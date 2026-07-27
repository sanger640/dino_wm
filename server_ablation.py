"""
Ablation server for the stability-centric safety monitor.
Supports multiple divergence metric modes for baselines and FTLE ablations.

Usage:
    python server_ablation.py mode=ftle                  # proposed method
    python server_ablation.py mode=final_cosine          # baseline: final state only
    python server_ablation.py mode=mean_traj             # baseline: mean over trajectory
    python server_ablation.py mode=max_step              # baseline: max single step
    python server_ablation.py mode=ftle_mean_patch       # ablation: mean all patches
    python server_ablation.py mode=ftle_gap              # ablation: global avg pool
    python server_ablation.py mode=ftle_l2               # ablation: L2 instead of cosine
    python server_ablation.py mode=ftle_topk             # ablation: top-K mean (K=5)
    python server_ablation.py mode=ftle_calibrated       # robustness: per-patch normalization
    python server_ablation.py mode=ftle_variance         # fix 1: cross-perturbation spread
    python server_ablation.py mode=ftle patch_stats_path=/path/to/patch_stats.npz
"""

import os
import zmq
import torch
import torch.nn.functional as F
import numpy as np
import hydra
import time
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms

CHECKPOINT_PATH = "/home/sanger/wksp/dino_wm/outputs/model_latest_single.pth"
PORT = 5556
TOP_K = 5

ALL_MODEL_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]


def load_model(model_ckpt, train_cfg, device):
    model_ckpt = Path(model_ckpt)

    def get_component(cfg_section=None, **kwargs):
        if cfg_section and hasattr(train_cfg, cfg_section):
            return hydra.utils.instantiate(getattr(train_cfg, cfg_section), **kwargs)
        return None

    kw = {}
    kw["encoder"] = get_component("encoder")
    emb_dim = getattr(kw["encoder"], "emb_dim", 384)
    kw["proprio_encoder"] = get_component("proprio_encoder", in_chans=4, emb_dim=train_cfg.proprio_emb_dim)
    kw["action_encoder"] = get_component("action_encoder", in_chans=4, emb_dim=train_cfg.action_emb_dim)

    concat_dim = getattr(train_cfg, "concat_dim", 0)
    patch_size = 16
    patches = (getattr(train_cfg, "img_size", 224) // patch_size) ** 2
    num_patches = patches + (2 if concat_dim == 0 else 0)
    pred_dim = emb_dim + (getattr(train_cfg, "action_emb_dim", 0) + getattr(train_cfg, "proprio_emb_dim", 0)) if concat_dim == 1 else emb_dim

    kw["predictor"] = get_component("predictor", dim=pred_dim, num_patches=num_patches, num_frames=train_cfg.num_hist)
    kw["decoder"] = get_component("decoder", emb_dim=emb_dim)
    kw.update({
        "proprio_dim": getattr(train_cfg, "proprio_emb_dim", 0),
        "action_dim": getattr(train_cfg, "action_emb_dim", 0),
        "concat_dim": concat_dim,
        "num_action_repeat": getattr(train_cfg, "num_action_repeat", 1),
        "num_proprio_repeat": getattr(train_cfg, "num_proprio_repeat", 1),
        "image_size": getattr(train_cfg, "img_size", 224),
        "num_hist": train_cfg.num_hist,
        "num_pred": train_cfg.num_pred,
    })

    model = hydra.utils.instantiate(train_cfg.model, **kw)
    model.to(device)

    if model_ckpt.exists():
        payload = torch.load(model_ckpt, map_location=device, weights_only=False)
        for key in ["encoder", "proprio_encoder", "action_encoder", "predictor", "decoder"]:
            module = getattr(model, key, None)
            if key in payload and module is not None:
                sd = payload[key] if isinstance(payload[key], dict) else payload[key].state_dict()
                module.load_state_dict(sd)
                print(f"  Loaded {key}")
    else:
        print(f"WARNING: checkpoint not found at {model_ckpt}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Divergence metric implementations
# All take z_orig (1, T, P, F) and z_noisy (B-1, T, P, F) and return (B-1,)
# n_hist: number of history steps (predicted steps start at index n_hist)
# ---------------------------------------------------------------------------

def _cosine_patch_dist(z_orig, z_noisy):
    """Per-patch cosine distance. Returns (B-1, T, P)."""
    return 1.0 - F.cosine_similarity(z_orig, z_noisy, dim=-1)


def _l2_patch_dist(z_orig, z_noisy):
    """Per-patch L2 distance. Returns (B-1, T, P)."""
    return torch.norm(z_orig - z_noisy, dim=-1)


def _collapse_patches(patch_dist, method):
    """Collapse (B-1, T, P) → (B-1, T) using patch aggregation method."""
    if method == "max":
        return patch_dist.amax(dim=-1)
    elif method == "mean":
        return patch_dist.mean(dim=-1)
    elif method == "topk":
        return torch.topk(patch_dist, k=TOP_K, dim=-1)[0].mean(dim=-1)
    elif method == "gap":
        gap = patch_dist.mean(dim=-1, keepdim=True)
        return gap.squeeze(-1)
    raise ValueError(f"Unknown patch method: {method}")


def _ftle(dist, n_hist):
    """FTLE from (B-1, T) distance trace → (B-1,)."""
    T_span = dist.shape[1] - n_hist
    d_start = dist[:, n_hist] + 1e-4
    d_end = dist[:, -1] + 1e-4
    return (1.0 / T_span) * torch.log(d_end / d_start)


MASK_TOP_ROWS = 28  # patches in top 2 rows (ceiling/background), same as server_single_max.py


def compute_score(z_orig, z_noisy, n_hist, mode, patch_stats=None):
    """
    Dispatches to the correct metric based on mode.
    Returns:
        scores (B-1,) numpy array — one score per perturbed trajectory
        patch_ftles (196,) numpy array — per-patch FTLE for worst trajectory (for calibration)
        worst_patch_idx (int)
    """
    # Patch-level FTLE (always computed for calibration output)
    patch_dist_cos = _cosine_patch_dist(z_orig, z_noisy)  # (B-1, T, 196)
    d_start_pp = patch_dist_cos[:, n_hist] + 1e-4          # (B-1, 196)
    d_end_pp = patch_dist_cos[:, -1] + 1e-4
    T_span = patch_dist_cos.shape[1] - n_hist
    lyap_per_patch = (1.0 / T_span) * torch.log(d_end_pp / d_start_pp)  # (B-1, 196)

    # Mask low-signal patches (noise floor) and top rows (ceiling background)
    lyap_masked = lyap_per_patch.clone()
    lyap_masked[d_end_pp < 1e-3] = -float("inf")
    lyap_masked[:, :MASK_TOP_ROWS] = -float("inf")

    # Per-trajectory max-patch FTLE (used for calibration output)
    patch_max_vals, patch_max_indices = torch.max(lyap_masked, dim=-1)  # (B-1,)

    # ---- MODE DISPATCH ----

    if mode == "ftle":
        # Proposed: per-patch FTLE, max patch, front camera
        scores = patch_max_vals

    elif mode == "ftle_calibrated":
        # Robustness fix 1: normalize per-patch FTLE by safe-trajectory baseline
        if patch_stats is None:
            raise ValueError("patch_stats required for ftle_calibrated mode")
        p95 = torch.from_numpy(patch_stats["p95"]).to(z_orig.device)  # (196,)
        normalized = lyap_masked / (p95.unsqueeze(0) + 1e-6)          # (B-1, 196)
        scores, patch_max_indices = torch.max(normalized, dim=-1)

    elif mode == "ftle_mean_patch":
        # Ablation: mean over all (unmasked) patches instead of max
        valid = lyap_per_patch.clone()
        valid[:, :MASK_TOP_ROWS] = float("nan")
        scores = torch.nanmean(valid, dim=-1)
        patch_max_indices = torch.zeros(valid.shape[0], dtype=torch.long, device=z_orig.device)

    elif mode == "ftle_topk":
        # Ablation: mean of top-K patches
        topk_vals, topk_idx = torch.topk(lyap_masked, k=TOP_K, dim=-1)  # (B-1, K)
        scores = topk_vals.mean(dim=-1)
        patch_max_indices = topk_idx[:, 0]

    elif mode == "ftle_gap":
        # Ablation: GAP features then FTLE (no per-patch)
        gap_orig = z_orig.mean(dim=-2)    # (1, T, F)
        gap_noisy = z_noisy.mean(dim=-2)  # (B-1, T, F)
        gap_dist = 1.0 - F.cosine_similarity(gap_orig, gap_noisy, dim=-1)  # (B-1, T)
        scores = _ftle(gap_dist, n_hist)
        patch_max_indices = torch.zeros(z_noisy.shape[0], dtype=torch.long, device=z_orig.device)

    elif mode == "ftle_l2":
        # Ablation: L2 norm instead of cosine
        l2_dist = _l2_patch_dist(z_orig, z_noisy)  # (B-1, T, P)
        l2_collapsed = _collapse_patches(l2_dist, "max")
        scores = _ftle(l2_collapsed, n_hist)
        patch_max_indices = torch.zeros(z_noisy.shape[0], dtype=torch.long, device=z_orig.device)

    elif mode == "final_cosine":
        # Baseline: final state cosine distance only (no FTLE ratio)
        final_dist = patch_dist_cos[:, -1, :]  # (B-1, 196)
        final_dist[:, :MASK_TOP_ROWS] = 0.0
        scores, patch_max_indices = torch.max(final_dist, dim=-1)

    elif mode == "mean_traj":
        # Baseline: mean cosine distance across predicted timesteps
        pred_dist = patch_dist_cos[:, n_hist:, :]  # (B-1, T_pred, 196)
        mean_dist = pred_dist.mean(dim=1)           # (B-1, 196)
        mean_dist[:, :MASK_TOP_ROWS] = 0.0
        scores, patch_max_indices = torch.max(mean_dist, dim=-1)

    elif mode == "max_step":
        # Baseline: max cosine distance at any single predicted timestep
        pred_dist = patch_dist_cos[:, n_hist:, :]  # (B-1, T_pred, 196)
        max_dist = pred_dist.amax(dim=1)            # (B-1, 196)
        max_dist[:, :MASK_TOP_ROWS] = 0.0
        scores, patch_max_indices = torch.max(max_dist, dim=-1)

    elif mode == "ftle_variance":
        # Fix 1: cross-perturbation spread of final latent states.
        # Measures how much the N perturbed rollouts diverge FROM EACH OTHER at the final
        # step — without using the original trajectory as a reference. This eliminates
        # the world-model prediction error on the original trajectory that inflates FTLE
        # for safe actions and forces the threshold above 0.
        #
        # For stable dynamics: all perturbed trajectories converge to similar endpoints
        # → spread ≈ 0. For unstable dynamics: some perturbations tip the block, others
        # don't → endpoints scatter widely → spread >> 0. Natural threshold ≈ 0.
        z_final = z_noisy[:, -1, :, :]               # (B-1, P, F) — final predicted step
        mu = z_final.mean(dim=0, keepdim=True)        # (1, P, F) — centroid across N perturbations
        # Per-perturbation, per-patch cosine distance from the centroid
        spread = 1.0 - F.cosine_similarity(
            z_final, mu.expand_as(z_final), dim=-1
        )                                              # (B-1, P)
        spread[:, :MASK_TOP_ROWS] = 0.0
        # Score per perturbation = how far it sits from the centroid at the worst patch
        scores, patch_max_indices = spread.max(dim=-1)  # (B-1,)

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: ftle, ftle_calibrated, "
                         "ftle_mean_patch, ftle_topk, ftle_gap, ftle_l2, "
                         "final_cosine, mean_traj, max_step, ftle_variance")

    scores_np = scores.cpu().numpy()
    indices_np = patch_max_indices.cpu().numpy()
    worst_j = int(np.argmax(scores_np))
    worst_patch_ftles = lyap_per_patch[worst_j].cpu().numpy()  # (196,) for calibration

    return scores_np, worst_patch_ftles, int(indices_np[worst_j])


@hydra.main(version_base=None, config_path="conf/", config_name="train")
def main(cfg: OmegaConf):
    mode = cfg.get("mode", "ftle")
    patch_stats_path = cfg.get("patch_stats_path", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        ckpt_path = Path(cfg.ckpt_base_path) / "outputs" / "model_latest.pth"

    print(f"Mode: {mode}")
    model = load_model(ckpt_path, cfg, device)
    TARGET_IMG_SIZE = getattr(cfg, "img_size", 224)

    # Normalization stats
    stats_path = Path(cfg.ckpt_base_path) / "outputs" / "dataset_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location=device)
        ACTION_MEAN, ACTION_STD = stats["action_mean"], stats["action_std"]
        PROPRIO_MEAN, PROPRIO_STD = stats["proprio_mean"], stats["proprio_std"]
    else:
        ACTION_MEAN = torch.tensor([0.45678952, 0.00051019, 0.50954217, 0.21926114], device=device)
        ACTION_STD  = torch.tensor([0.03182372, 0.01151787, 0.03419121, 0.41397065], device=device)
        PROPRIO_MEAN = torch.tensor([0.4564166, 0.00056233, 0.50817657, 0.21921302], device=device)
        PROPRIO_STD  = torch.tensor([0.03217997, 0.01056713, 0.0327194,  0.4139551 ], device=device)

    # Per-patch calibration stats (for ftle_calibrated mode)
    patch_stats = None
    if mode == "ftle_calibrated":
        if patch_stats_path is None:
            patch_stats_path = str(Path(cfg.ckpt_base_path) / "outputs" / "patch_stats.npz")
        patch_stats = np.load(patch_stats_path)
        print(f"Loaded patch stats from {patch_stats_path}")

    inference_transform = transforms.Compose([
        transforms.Resize(TARGET_IMG_SIZE),
        transforms.CenterCrop(TARGET_IMG_SIZE),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"Ablation server [{mode}] listening on tcp://*:{PORT}")

    while True:
        try:
            message = socket.recv_pyobj()
            t0 = time.time()

            def to_tensor(arr):
                t = torch.from_numpy(arr).float().to(device)
                if arr.dtype == np.uint8:
                    t = t / 255.0
                return t

            visual_t = to_tensor(message["visual"])
            proprio_t = to_tensor(message["proprio"])
            actions_t = to_tensor(message["actions"])

            if visual_t.ndim == 4: visual_t = visual_t.unsqueeze(0)
            if proprio_t.ndim == 2: proprio_t = proprio_t.unsqueeze(0)
            if actions_t.ndim == 2: actions_t = actions_t.unsqueeze(0)

            proprio_t = (proprio_t - PROPRIO_MEAN) / PROPRIO_STD
            actions_t = (actions_t - ACTION_MEAN) / ACTION_STD

            b, t, c, h, w = visual_t.shape
            visual_t = inference_transform(visual_t.view(b * t, c, h, w)).view(b, t, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)

            n_hist = visual_t.shape[1]
            b_size = actions_t.shape[0]

            with torch.no_grad():
                z_obses, _ = model.rollout({"visual": visual_t, "proprio": proprio_t}, actions_t)

            scores_np = np.zeros(max(1, b_size - 1))
            worst_patch_ftles = np.zeros(196)
            worst_patch_idx = 0
            max_score = 0.0

            if b_size > 1 and z_obses["visual"].shape[1] > n_hist:
                z_visual = z_obses["visual"]           # (B, T, 196, 384)
                z_orig  = z_visual[0:1]                # (1, T, 196, 384)
                z_noisy = z_visual[1:]                 # (B-1, T, 196, 384)

                scores_np, worst_patch_ftles, worst_patch_idx = compute_score(
                    z_orig, z_noisy, n_hist, mode, patch_stats
                )
                max_score = float(np.max(scores_np))

            # Decode only original + worst trajectory (if decoder available)
            decoded_images = None
            if hasattr(model, "decoder") and model.decoder is not None and b_size > 1:
                worst_j = int(np.argmax(scores_np))
                z_subset = {k: v[[0, worst_j + 1]] for k, v in z_obses.items()}
                with torch.no_grad():
                    decoded_obs, _ = model.decode_obs(z_subset)
                pred_np = (decoded_obs["visual"].cpu().numpy() + 1.0) / 2.0
                decoded_images = np.clip(pred_np * 255, 0, 255).astype(np.uint8)

            socket.send_pyobj({
                "states":           decoded_images,
                "max_lyapunov":     max_score,
                "max_patch_idx":    worst_patch_idx,
                "all_lyapunovs":    scores_np,
                "worst_patch_ftles": worst_patch_ftles,  # (196,) for calibration
                "mode":             mode,
                "inference_time":   time.time() - t0,
            })

        except Exception as e:
            print(f"Error: {e}")
            try:
                socket.send_pyobj({"error": str(e)})
            except:
                pass


if __name__ == "__main__":
    main()
