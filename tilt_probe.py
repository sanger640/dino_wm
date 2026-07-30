"""
Is block tilt linearly decodable from the latents -- and from the PREDICTED ones?

Freeze the representation, fit a linear (ridge) readout to ground-truth tilt, score it on
held-out episodes. Linear is the point: it asks whether the information is present AND
directly usable by a downstream distance, not whether a deep head could dig it out.

Two readouts, and the difference between them is the whole experiment:

  ENCODER   tilt at time t from the real DINOv2 features of the frame at time t.
            "Does the representation see tilt at all?" Near-zero here would mean the
            frozen encoder discards it and no world-model change can help.

  PREDICTOR tilt at time t+8 from the latent the world model PREDICTS for t+8, given
            only frames up to t. "Does the model know the block is about to tip?"
            This is the quantity the safety monitor actually needs.

The comparison that matters is PREDICTOR-probe AUC against d_end's AUC on the same chunks.
If the probe wins clearly, the world model holds tilt information that the divergence
metric is throwing away -- which converts "the model works, the signal is wrong" from a
hypothesis into a measurement, and bounds what a better readout could achieve.

Nothing here designates a "tilt feature". Ridge is handed every dimension and finds the
direction w itself; w is the OUTPUT. Two things decide whether that output means anything:

  POOLING.    Tilt occupies a couple of patches out of 84, so mean-pooling can bury it and
              make "no signal" indistinguishable from "destroyed by averaging". Because the
              front camera is FIXED, patch index p refers to the same scene region in every
              episode, so spatial layout is stable and directly usable. Three readouts are
              scored: mean-pool, per-patch (max-aggregated, which also yields a heatmap),
              and full concatenation of the kept patches.

  CONFOUNDS.  Tilt rises late in an episode and while the arm is near the blocks, so a
              probe could score well by decoding elapsed time or arm pose and never look at
              a block. Two null baselines -- tilt from timestep alone, tilt from proprio
              alone -- must be beaten before any latent result counts.

Requires an LMDB built from episodes recorded with per-step tilt (replay_noisy.py writes
tilt_left/tilt_right; create_lmdb_single30.py aligns them into <ep>_tilt). It refuses to
run against older data rather than silently probing zeros.
"""
import argparse, json, pickle
from pathlib import Path

import cv2, hydra, lmdb, numpy as np, torch

from server_single_max import load_model, build_patch_keep_mask
from torchvision import transforms

PM = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PSD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]
AM = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ASD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
NH, NP = 3, 8
dev = "cuda"


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def auc(p, n):
    p = np.asarray(p, float); n = np.asarray(n, float)
    p = p[np.isfinite(p)]; n = np.sort(n[np.isfinite(n)])
    if not len(p) or not len(n):
        return float("nan")
    r = np.searchsorted(n, p, "left") + 0.5 * (
        np.searchsorted(n, p, "right") - np.searchsorted(n, p, "left"))
    return float(r.mean() / len(n))


def ridge_fit(X, y, lam):
    Xb = np.c_[X, np.ones(len(X))]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def ridge_apply(X, w):
    return np.c_[X, np.ones(len(X))] @ w


def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/tilt_probe.json")
    args = ap.parse_args()
    N = args.n_perturb

    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_model(Path("outputs/model_latest_single.pth"), cfg, dev); model.eval()
    tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)
    am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
    keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
    span = NH + NP
    cs = torch.nn.functional.cosine_similarity
    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    rows = []

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = list(meta["episodes"])
        if txn.get(f"{eps[0]}_tilt".encode()) is None:
            raise SystemExit(
                f"{args.lmdb} carries no <ep>_tilt key.\n"
                "Rebuild from episodes recorded by the tilt-logging replay_noisy.py; "
                "probing without it would silently fit noise.")
        print(f"{len(eps)} episodes with per-step tilt")

        for ei, ep in enumerate(eps):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))       # (T, 2) left/right
            n = min(len(keys), len(acts), len(props), len(tilt))
            for s in range(0, n - span, NP):
                raw = [txn.get(keys[s + j].encode()) for j in range(span)]
                if any(r is None for r in raw):
                    break
                vis = tf(torch.from_numpy(
                    np.stack([np.transpose(dec(r), (2, 0, 1)) for r in raw])).float().to(dev) / 255.)
                g = torch.Generator(device=dev); g.manual_seed(s)
                a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
                a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
                obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                       "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                                   ).unsqueeze(0).repeat(N, 1, 1)}
                with torch.no_grad():
                    z, _ = model.rollout(obs, (a - am) / asd)
                zv = z["visual"]
                de = ((1 - cs(zv[1:, -1], zv[0:1, -1], dim=-1)) + 1e-4).mean(0).cpu().numpy()
                rows.append({
                    "ep": ep, "step": s,
                    # full per-patch features; pooling happens at probe time so all three
                    # readouts see identical latents
                    "Z_obs": zv[0, NH - 1][keep].cpu().numpy().astype(np.float32),
                    "Z_pred": zv[0, -1][keep].cpu().numpy().astype(np.float32),
                    "proprio": props[s + NH - 1].astype(np.float32),
                    "tilt_now": float(tilt[s + NH - 1].max()),
                    "tilt_fut": float(tilt[min(s + span - 1, len(tilt) - 1)].max()),
                    "dend_p90": float(np.percentile(de[keep], 90)),
                })
            if ei % 10 == 0:
                print(f"  [{ei}/{len(eps)}] chunks={len(rows)}", flush=True)
    env.close()

    eps_u = sorted({r["ep"] for r in rows})
    cut = set(eps_u[:len(eps_u) // 2])
    tr = [r for r in rows if r["ep"] in cut]
    te = [r for r in rows if r["ep"] not in cut]
    print(f"\n{len(rows)} chunks | train {len(tr)} | held-out {len(te)}")
    print(f"tilt range: now {min(r['tilt_now'] for r in rows):.1f}-"
          f"{max(r['tilt_now'] for r in rows):.1f} deg, "
          f"future {min(r['tilt_fut'] for r in rows):.1f}-"
          f"{max(r['tilt_fut'] for r in rows):.1f} deg")

    def build(rs, feat, mode):
        if feat in ("Z_obs", "Z_pred"):
            Z = np.stack([r[feat] for r in rs])                  # (n, P, F)
            if mode == "pool":
                return Z.mean(1)
            if mode == "concat":
                return Z.reshape(len(Z), -1)
            return Z                                             # per-patch, handled below
        if feat == "step":
            return np.array([[r["step"]] for r in rs], dtype=np.float32)
        return np.stack([r["proprio"] for r in rs])

    def run(name, feat, targ, mode):
        ytr = np.array([r[targ] for r in tr]); yte = np.array([r[targ] for r in te])
        if mode == "patch":
            # one shared readout applied at every patch, aggregated by max: the scene-level
            # target is attached to each patch, so this is a deliberately crude MIL-style
            # fit -- its value is the heatmap and its robustness to pooling dilution
            Ztr, Zte = build(tr, feat, mode), build(te, feat, mode)
            X = Ztr.reshape(-1, Ztr.shape[-1])
            y = np.repeat(ytr, Ztr.shape[1])
            w = ridge_fit(X, y, args.lam * Ztr.shape[1])
            pred = ridge_apply(Zte.reshape(-1, Zte.shape[-1]), w
                               ).reshape(len(Zte), -1).max(1)
        else:
            Xtr, Xte = build(tr, feat, mode), build(te, feat, mode)
            lam = args.lam * (Xtr.shape[1] / 384.0)
            w = ridge_fit(Xtr, ytr, lam); pred = ridge_apply(Xte, w)
        a10 = auc(pred[yte > 10], pred[yte <= 10])
        a45 = auc(pred[yte > 45], pred[yte <= 45])
        cc = np.corrcoef(yte, pred)[0, 1] if np.std(pred) > 0 else np.nan
        print(f"{name:<40}{r2(yte, pred):>8.3f}{cc:>8.3f}{a10:>11.3f}{a45:>11.3f}")
        return {"r2": r2(yte, pred), "auc10": a10, "auc45": a45}

    print(f"\n{'probe':<40}{'R2':>8}{'corr':>8}{'AUC>10deg':>11}{'AUC>45deg':>11}")
    print("-" * 78)
    results = {}
    print("--- null baselines: beat these or the latent result means nothing ---")
    results["null_step->tilt_fut"] = run("timestep only     -> tilt_t+8", "step", "tilt_fut", "pool")
    results["null_proprio->tilt_fut"] = run("proprio only      -> tilt_t+8", "prop", "tilt_fut", "pool")
    print("--- encoder: is tilt visible in the real DINOv2 features? ---")
    for mode in ("pool", "patch", "concat"):
        results[f"enc_{mode}"] = run(f"ENCODER  z_t [{mode}]   -> tilt_t", "Z_obs", "tilt_now", mode)
    print("--- predictor: does the model know tilt 8 steps ahead? ---")
    for mode in ("pool", "patch", "concat"):
        results[f"pred_{mode}"] = run(f"PREDICTOR z_t+8 [{mode}] -> tilt_t+8", "Z_pred", "tilt_fut", mode)

    # the reference line: the metric currently in use, on the SAME held-out chunks
    dv = np.array([r["dend_p90"] for r in te]); yf = np.array([r["tilt_fut"] for r in te])
    print(f"{'d_end p90 (current metric)':<34}{'-':>8}"
          f"{np.corrcoef(yf, dv)[0,1]:>8.3f}"
          f"{auc(dv[yf>10], dv[yf<=10]):>11.3f}{auc(dv[yf>45], dv[yf<=45]):>11.3f}")
    print("\nIf the PREDICTOR probe beats d_end, the world model holds tilt information that")
    print("the divergence metric discards, and the gap bounds what a better readout can win.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
