"""
Two rigorous tests of the world model, in latent space, with no decoder involved.

Decoded pixels are a lossy view: the VQVAE could blur out a topple that the latents encode
perfectly well, or vice versa. Both tests below compare latents directly.

TEST A -- do actions drive the prediction?
    Roll out the SAME observation history under several different action sequences (the true
    actions, and deliberately altered ones) and measure how far the predicted final latents
    move apart. If altering the actions barely changes the prediction, the predictor is
    ignoring them. Reported as cosine distance between final latents, alongside the
    same-input repeat distance as a noise floor.

TEST B -- does the model predict what actually happens?
    Encode the ground-truth future frames and compare them against the model's predicted
    latents at the same timesteps. Then split by outcome: if the model captures toppling,
    its error on failure episodes near the topple should be comparable to its error on
    success episodes. If it systematically misses topples, error will be markedly higher
    there, and concentrated in the patches where the block actually moved.

Usage:
    python latent_probe.py --lmdb ../panda_express/tasks/jenga_noise_50/jenga_single.lmdb \
        --labels ../panda_express/labels_noise50.json
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import hydra
import lmdb
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms

from server_single_max import load_model

ACTION_MEAN = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ACTION_STD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PROPRIO_MEAN = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PROPRIO_STD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--ckpt", default="/home/sanger/wksp/dino_wm/outputs/model_latest_single.pth")
    p.add_argument("--num-hist", type=int, default=3)
    p.add_argument("--num-pred", type=int, default=8)
    p.add_argument("--offset", type=int, default=-8, help="chunk start relative to failure_step")
    p.add_argument("--max-success", type=int, default=12)
    p.add_argument("--mask-top", type=int, default=28)
    p.add_argument("--output", default="results/latent_probe.json")
    return p.parse_args()


def decode_img(b):
    a = np.frombuffer(b, dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def cos_dist(a, b):
    """Per-patch cosine distance between (P,D) latents."""
    return (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_model(Path(args.ckpt), cfg, device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.CenterCrop(cfg.img_size),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    am = torch.tensor(ACTION_MEAN, device=device); asd = torch.tensor(ACTION_STD, device=device)
    pm = torch.tensor(PROPRIO_MEAN, device=device); psd = torch.tensor(PROPRIO_STD, device=device)

    labels = json.load(open(args.labels))
    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    NH, NP = args.num_hist, args.num_pred
    span = NH + NP

    resA, resB = [], []
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))

        # choose chunks: every failure episode near its topple, plus success episodes mid-episode
        targets = []
        for ep, v in labels.items():
            if ep not in meta["episodes"]:
                continue
            keys = meta["episodes"][ep]["keys"]["cam2"]
            if v["outcome"] == "failure":
                s = v["failure_step"] + args.offset
                if 0 <= s and s + span < len(keys):
                    targets.append((ep, s, "failure"))
            elif len([t for t in targets if t[2] == "success"]) < args.max_success:
                s = len(keys) // 2
                if s + span < len(keys):
                    targets.append((ep, s, "success"))

        for ep, start, kind in targets:
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))

            raw = [decode_img(txn.get(keys[start + t].encode())) for t in range(span)]
            vis = torch.from_numpy(np.stack([np.transpose(r, (2, 0, 1)) for r in raw])).float().to(device) / 255.0
            vis = tf(vis).unsqueeze(0)                                   # (1, span, 3, H, W)
            pro = ((torch.from_numpy(props[start:start + span]).float().to(device) - pm) / psd).unsqueeze(0)
            act = ((torch.from_numpy(acts[start:start + span]).float().to(device) - am) / asd).unsqueeze(0)

            with torch.no_grad():
                # ---------- TEST B: predicted vs actual future latents ----------
                obs0 = {"visual": vis[:, :NH], "proprio": pro[:, :NH]}
                z_pred, _ = model.rollout(obs0, act)
                z_gt = model.encode_obs({"visual": vis, "proprio": pro})["visual"][0]   # (span,P,D)
                zp = z_pred["visual"][0]                                                # (T,P,D)
                T = min(zp.shape[0], z_gt.shape[0])

                errs = []
                for t in range(NH, T):
                    d = cos_dist(zp[t], z_gt[t])
                    d[:args.mask_top] = float("nan")
                    errs.append(float(torch.nanmean(d)))
                # where is the error concentrated at the final step?
                dfin = cos_dist(zp[T - 1], z_gt[T - 1]); dfin[:args.mask_top] = float("nan")
                topk = torch.topk(torch.nan_to_num(dfin, nan=-1), 10).indices.tolist()
                resB.append(dict(ep=ep, kind=kind, start=start,
                                 err_first=errs[0], err_last=errs[-1],
                                 err_mean=float(np.mean(errs)), top_patches=topk))

                # ---------- TEST A: action sensitivity ----------
                variants = {
                    "same": act.clone(),
                    "small_5mm": act.clone(),
                    "large_2cm": act.clone(),
                    "frozen": act.clone(),
                }
                variants["small_5mm"][..., :3] += (0.005 / torch.tensor(ACTION_STD[:3], device=device))
                variants["large_2cm"][..., :3] += (0.02 / torch.tensor(ACTION_STD[:3], device=device))
                variants["frozen"][:, NH:, :3] = variants["frozen"][:, NH - 1:NH, :3]
                finals = {}
                for name, a in variants.items():
                    zz, _ = model.rollout(obs0, a)
                    finals[name] = zz["visual"][0][-1]
                base = finals["same"]
                row = dict(ep=ep, kind=kind)
                for name in ["small_5mm", "large_2cm", "frozen"]:
                    d = cos_dist(finals[name], base); d[:args.mask_top] = float("nan")
                    row[name] = float(torch.nanmean(d))
                resA.append(row)

            print(f"  {ep:>4} {kind:>8} start {start:>4} | pred-vs-actual err "
                  f"first {resB[-1]['err_first']:.4f} last {resB[-1]['err_last']:.4f} | "
                  f"action sens 2cm {resA[-1]['large_2cm']:.4f} frozen {resA[-1]['frozen']:.4f}")

    env.close()

    def agg(rows, key, kind):
        v = [r[key] for r in rows if r["kind"] == kind]
        return (np.mean(v), np.std(v), len(v)) if v else (float("nan"),) * 3

    print("\n================ TEST A: do actions drive the prediction? ================")
    print(f"{'variant':>12} | {'failure chunks':>22} | {'success chunks':>22}")
    for k in ["small_5mm", "large_2cm", "frozen"]:
        mf, sf, nf = agg(resA, k, "failure"); ms, ss, ns = agg(resA, k, "success")
        print(f"{k:>12} | {mf:>10.5f} +- {sf:<8.5f} | {ms:>10.5f} +- {ss:<8.5f}")
    print("  (cosine distance of final latent vs the true-action rollout;")
    print("   ~0 would mean the predictor ignores the action input)")

    print("\n========== TEST B: does the prediction match what actually happened? ==========")
    for k in ["err_first", "err_last", "err_mean"]:
        mf, sf, nf = agg(resB, k, "failure"); ms, ss, ns = agg(resB, k, "success")
        print(f"{k:>10} | failure {mf:.4f} +- {sf:.4f} (n={nf}) | success {ms:.4f} +- {ss:.4f} (n={ns})")
    mf, _, _ = agg(resB, "err_last", "failure"); ms, _, _ = agg(resB, "err_last", "success")
    print(f"\n  failure/success final-step error ratio = {mf/ms:.2f}")
    print("  ~1.0 => model predicts topples as well as it predicts safe futures")
    print("  >>1  => model specifically fails where the topple happens")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"test_a": resA, "test_b": resB}, open(args.output, "w"), indent=2)
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
