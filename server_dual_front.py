"""
Dual-camera safety monitor: both views feed the world model, divergence is measured on the
FIXED camera only.

The dual checkpoint (model_latest.pth, epoch 46) uses models.dual_visual_world_model.VWorldModel,
which differs from the single-view model in three ways that matter here:

  * two views are concatenated along the PATCH dimension, 392 patches per frame
  * predict() routes them through separate MLP heads, and the ordering is explicit in that
    code: "# Tokens: [Wrist Patches, Front Patches]", so

        patches   0-195  = wrist  (cam1)
        patches 196-391  = front / fixed  (cam2)

  * it has decoder_front / decoder_wrist rather than a single decoder

Scoring uses ONLY the front half, with the same row mask applied within that view
(rows 2-7 of the 14x14 grid), giving patch indices 224-307. The wrist camera still informs
the prediction through cross-view attention in the ViT -- it just does not contribute to
the divergence measurement.

    python server_dual_front.py                 # front-only divergence (default)
    SCORE_VIEW=wrist python server_dual_front.py
    SCORE_VIEW=both  python server_dual_front.py

Listens on port 5557 so it can run alongside the single-view server on 5556.
"""
import os
import time

import hydra
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf
from pathlib import Path
from torchvision import transforms

CHECKPOINT_PATH = os.environ.get("DUAL_CKPT", "/home/sanger/Downloads/model_latest.pth")
PORT = int(os.environ.get("PORT", 5557))
SCORE_VIEW = os.environ.get("SCORE_VIEW", "front")   # front | wrist | both

PATCH_GRID = 14
PATCHES_PER_VIEW = PATCH_GRID * PATCH_GRID           # 196
MASKED_ROWS = (0, 1, 8, 9, 10, 11, 12, 13)           # arm/upper background + checkered floor

ACTION_MEAN = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ACTION_STD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PROPRIO_MEAN = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PROPRIO_STD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]


def build_keep_mask(num_patches, device):
    """True where a patch may contribute to the divergence score."""
    keep = torch.zeros(num_patches, dtype=torch.bool, device=device)
    n_views = max(1, num_patches // PATCHES_PER_VIEW)
    views = {"wrist": [0], "front": [1], "both": list(range(n_views))}[SCORE_VIEW]
    for v in views:
        if v >= n_views:
            continue
        base = v * PATCHES_PER_VIEW
        block = torch.ones(PATCHES_PER_VIEW, dtype=torch.bool, device=device)
        for r in MASKED_ROWS:
            block[r * PATCH_GRID:(r + 1) * PATCH_GRID] = False
        keep[base:base + PATCHES_PER_VIEW] = block
    return keep


def load_dual_model(ckpt_path, cfg, device):
    def comp(section, **kw):
        return hydra.utils.instantiate(getattr(cfg, section), **kw) if hasattr(cfg, section) else None

    encoder = comp("encoder")
    emb = getattr(encoder, "emb_dim", 384)
    kw = dict(
        encoder=encoder,
        proprio_encoder=comp("proprio_encoder", in_chans=4, emb_dim=cfg.proprio_emb_dim),
        action_encoder=comp("action_encoder", in_chans=4, emb_dim=cfg.action_emb_dim),
        decoder_front=comp("decoder", emb_dim=emb),
        decoder_wrist=comp("decoder", emb_dim=emb),
    )
    concat_dim = getattr(cfg, "concat_dim", 1)
    num_patches = PATCHES_PER_VIEW * 2 + (2 if concat_dim == 0 else 0)
    dim = emb + (getattr(cfg, "action_emb_dim", 0) + getattr(cfg, "proprio_emb_dim", 0)) if concat_dim == 1 else emb
    kw["predictor"] = comp("predictor", dim=dim, num_patches=num_patches, num_frames=cfg.num_hist)
    kw.update(dict(proprio_dim=getattr(cfg, "proprio_emb_dim", 0),
                   action_dim=getattr(cfg, "action_emb_dim", 0),
                   concat_dim=concat_dim,
                   num_action_repeat=getattr(cfg, "num_action_repeat", 1),
                   num_proprio_repeat=getattr(cfg, "num_proprio_repeat", 1),
                   image_size=getattr(cfg, "img_size", 224),
                   num_hist=cfg.num_hist, num_pred=cfg.num_pred))
    model = hydra.utils.instantiate(cfg.model, **kw).to(device)

    payload = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    print(f"checkpoint epoch {payload.get('epoch')}, keys: {[k for k in payload if 'optimizer' not in k]}")
    mapping = {"predictor": getattr(model, "predictor", None),
               "decoder_front": getattr(model, "decoder_front", None),
               "decoder_wrist": getattr(model, "decoder_wrist", None),
               "proprio_encoder": model.proprio_encoder,
               "action_encoder": model.action_encoder,
               "wrist_head": getattr(model, "wrist_head", None),
               "front_head": getattr(model, "front_head", None),
               "proprio_head": getattr(model, "proprio_head", None)}
    for k, mod in mapping.items():
        if k in payload and mod is not None:
            obj = payload[k]
            mod.load_state_dict(obj if isinstance(obj, dict) else obj.state_dict())
            print(f"  loaded {k}")
        elif mod is not None:
            print(f"  MISSING in checkpoint: {k}")
    model.eval()
    return model


def score_all(z_noisy, z_orig, n_hist, T_span, keep):
    """Every candidate statistic, restricted to `keep` patches."""
    d = 1 - torch.nn.functional.cosine_similarity(z_noisy, z_orig, dim=-1)
    ds = d[:, n_hist] + 1e-4
    de = d[:, -1] + 1e-4
    de_k, ds_k = de[:, keep], ds[:, keep]
    lam = (1.0 / T_span) * torch.log(de_k / ds_k)
    floor = de_k > 1e-3
    lam_f = torch.where(floor, lam, torch.full_like(lam, -float("inf")))
    per_pert = lam_f.max(dim=-1).values
    fin = torch.isfinite(per_pert)
    lam_var = (1.0 / T_span) * torch.log((de_k.std(dim=0) + 1e-6) / (ds_k.std(dim=0) + 1e-6))

    def f(x):
        x = float(x)
        return x if np.isfinite(x) else float("nan")

    out = {
        "dend_std": f(de_k.std(dim=0).mean()),
        "dend_mean": f(de_k.mean()),
        "dend_p90": f(torch.quantile(de_k.flatten().float(), 0.90)),
        "dend_max": f(de_k.max()),
        "dend_maxpatch_meanpert": f(de_k.max(dim=-1).values.mean()),
        "dend_meanpatch_maxpert": f(de_k.mean(dim=-1).max()),
        "ddiff_mean": f((de_k - ds_k).mean()),
        "ftle": f(per_pert[fin].max() if fin.any() else float("nan")),
        "ftle_maxpatch_meanpert": f(per_pert[fin].mean() if fin.any() else float("nan")),
        "ftle_maxpatch_medpert": f(per_pert[fin].median() if fin.any() else float("nan")),
        "ftle_mean": f(lam.mean()),
        "lambda_var_meanpatch": f(lam_var.mean()),
        "lambda_var_medpatch": f(lam_var.median()),
        "lambda_var_maxpatch": f(lam_var.max()),
    }
    spread = de.std(dim=0)
    spread = torch.where(keep, spread, torch.zeros_like(spread))
    return out, int(torch.argmax(spread)), int(torch.argmax(de[:, int(torch.argmax(spread))]))


@hydra.main(version_base=None, config_path="conf/", config_name="serve_dual")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml({"img_size": cfg.img_size, "num_hist": cfg.num_hist,
                             "concat_dim": cfg.concat_dim}))
    model = load_dual_model(CHECKPOINT_PATH, cfg, device)

    size = getattr(cfg, "img_size", 224)
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    am = torch.tensor(ACTION_MEAN, device=device); asd = torch.tensor(ACTION_STD, device=device)
    pm = torch.tensor(PROPRIO_MEAN, device=device); psd = torch.tensor(PROPRIO_STD, device=device)

    ctx = zmq.Context(); sock = ctx.socket(zmq.REP); sock.bind(f"tcp://*:{PORT}")
    print(f"Dual-view server listening on tcp://*:{PORT}  |  scoring view: {SCORE_VIEW.upper()}")

    keep_cache = {}
    while True:
        msg = sock.recv_pyobj()
        t0 = time.time()
        try:
            vis = torch.from_numpy(msg["visual"]).float().to(device)
            if vis.dtype == torch.float32 and msg["visual"].dtype == np.uint8:
                vis = vis / 255.0
            pro = torch.from_numpy(msg["proprio"]).float().to(device)
            act = torch.from_numpy(msg["actions"]).float().to(device)
            pro = (pro - pm) / psd
            act = (act - am) / asd

            b, t, v, c, h, w = vis.shape                      # (B, T, 2, C, H, W)
            vis = tf(vis.view(b * t * v, c, h, w)).view(b, t, v, c, size, size)

            with torch.no_grad():
                z_obs, _ = model.rollout({"visual": vis, "proprio": pro}, act)
                zv = z_obs["visual"]
                n_hist = vis.shape[1]
                key = zv.shape[-2]
                if key not in keep_cache:
                    keep_cache[key] = build_keep_mask(key, zv.device)
                    print(f"  scoring {int(keep_cache[key].sum())}/{key} patches "
                          f"(view={SCORE_VIEW})")
                keep = keep_cache[key]
                scores, patch_idx, worst = score_all(zv[1:], zv[0:1], n_hist,
                                                     zv.shape[1] - n_hist, keep)
            sock.send_pyobj({"max_lyapunov": scores["dend_mean"],
                             "alt_scores": scores,
                             "max_patch_idx": patch_idx,
                             "all_lyapunovs": np.zeros(1),
                             "states": None,
                             "inference_time": time.time() - t0})
        except Exception as e:
            import traceback; traceback.print_exc()
            sock.send_pyobj({"error": str(e)})


if __name__ == "__main__":
    main()
