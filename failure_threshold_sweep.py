"""
Does the monitor resolve disturbance BELOW the topple threshold?

Peak adjacent-block tilt over 100 episodes is starkly bimodal: 70 episodes under 17.5 deg,
30 episodes at 90-100 deg, and ZERO in between. So moving the failure threshold anywhere in
20-90 deg leaves the labels unchanged -- the 45 deg cut sits in an empty gap and is already
about as robust as a binary choice can be.

The only redefinition that changes anything cuts INSIDE the standing mode (5.2-16.8 deg,
median 11.1). That asks a different question than "did it topple": does the monitor rank
moments of larger disturbance above smaller ones, when nothing actually falls?

Sweeping the tilt threshold that defines "unsafe" from 8 deg (well inside normal wobble) up
to 45 deg (the topple definition) and recomputing labels from scratch at each level:

    fstep(X)  = first step where either adjacent block exceeds X
    positive  = fstep(X) falls in this chunk's 8-step prediction horizon
    dropped   = chunks after fstep(X)      (aftermath, not prediction)

Every metric is scored on identical rollouts; only the labels move. If AUC holds up at low
thresholds, the monitor is reading a continuum of instability that the binary topple label
discards. If it collapses toward chance, the monitor only sees actual falls and the binary
label is the right one.
"""
import argparse, json, pickle
from pathlib import Path

import cv2, hydra, lmdb, numpy as np, torch

from server_single_max import load_model, build_patch_keep_mask
from torchvision import transforms

AM = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ASD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PM = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PSD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]
NH, NP = 3, 8
dev = "cuda"
LOWNORM_K = 30
FLOOR = 1e-3
THRESHOLDS = [8.0, 10.0, 12.0, 15.0, 20.0, 45.0]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/sanger/wksp/panda_express/tasks/jenga_tilt_100/jenga_tilt.lmdb")
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-fit-episodes", type=int, default=50)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/failure_threshold_sweep.json")
    args = ap.parse_args()
    N = args.n_perturb

    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_model(Path("outputs/model_latest_single.pth"), cfg, dev); model.eval()
    tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
    pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)
    keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
    span = NH + NP
    cs = torch.nn.functional.cosine_similarity
    env = lmdb.open(args.lmdb, readonly=True, lock=False)

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = sorted(meta["episodes"], key=lambda e: int(e))
        fit_eps, held = eps[:args.n_fit_episodes], eps[args.n_fit_episodes:]

        def load(ep):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            return keys, acts, props, tilt, min(len(keys), len(acts), len(props), len(tilt))

        def roll(ep, s, keys, acts, props, n_pert):
            raw = [txn.get(keys[s + j].encode()) for j in range(span)]
            if any(r is None for r in raw):
                return None
            vis = tf(torch.from_numpy(np.stack([np.transpose(dec(r), (2, 0, 1)) for r in raw])
                                      ).float().to(dev) / 255.)
            g = torch.Generator(device=dev); g.manual_seed(s)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(n_pert, 1, 1)
            if n_pert > 1:
                a[1:, :, :3] += torch.randn(n_pert - 1, span, 3, device=dev,
                                            generator=g) * args.noise_std
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(n_pert, 1, 1, 1, 1),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                               ).unsqueeze(0).repeat(n_pert, 1, 1)}
            with torch.no_grad():
                z, _ = model.rollout(obs, (a - am) / asd)
            return z["visual"]

        print(f"fitting probe on {len(fit_eps)} episodes (N=1)", flush=True)
        X, Y = [], []
        for ep in fit_eps:
            keys, acts, props, tilt, n = load(ep)
            for s in range(0, n - span, NP):
                zv = roll(ep, s, keys, acts, props, 1)
                if zv is None:
                    break
                X.append(zv[0, -1][keep].mean(0).cpu().numpy())
                Y.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
        w = ridge_fit(np.stack(X), np.array(Y), args.lam)
        print(f"  fit on {len(X)} chunks")

        # score EVERY chunk of the held-out half; labels are applied later, per threshold
        print(f"scoring all chunks of {len(held)} held-out episodes (N={N})", flush=True)
        rows = []
        for i, ep in enumerate(held):
            keys, acts, props, tilt, n = load(ep)
            for s in range(0, n - span, NP):
                zv = roll(ep, s, keys, acts, props, N)
                if zv is None:
                    break
                zo, zn = zv[0:1], zv[1:]
                ds = ((1 - cs(zn[:, NH], zo[:, NH], dim=-1)) + 1e-4).cpu().numpy()
                de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).cpu().numpy()
                nrm = zo[0, NH - 1].norm(dim=-1).cpu().numpy()
                m = keep.copy(); m[np.argsort(nrm)[:LOWNORM_K]] = False
                lp = np.where(de > FLOOR, (1.0 / NP) * np.log(de / np.median(ds[:, m])), np.nan)
                with np.errstate(invalid="ignore"):
                    lp_p = np.nanmean(lp, axis=0)
                rows.append({
                    "ep": ep, "s": s,
                    "probe": float(np.c_[zv[0, -1][keep].mean(0).cpu().numpy()[None],
                                         np.ones((1, 1))] @ w),
                    "d_end": float(np.percentile(de[:, m].mean(0), 90)),
                    "ftle_pooled": float(np.nanpercentile(lp_p[m], 90)),
                })
            if i % 10 == 0:
                print(f"  [{i}/{len(held)}] chunks={len(rows)}", flush=True)

        tilts = {}
        for ep in held:
            tilts[ep] = load(ep)[3]
    env.close()

    METS = ["probe", "d_end", "ftle_pooled"]
    print(f"\n{len(rows)} chunks scored across {len(held)} held-out episodes")
    print("\n=== AUC as the failure threshold moves INTO the standing mode ===")
    print("  standing mode is 5.2-16.8 deg (median 11.1); 20-90 deg is empty, so any")
    print("  threshold in that range gives identical labels to 45 deg")
    hdr = (f"{'thr (deg)':>10}{'n_pos':>7}{'n_neg':>7}{'base %':>8}"
           + "".join(f"{m:>14}" for m in METS))
    print(hdr); print("-" * len(hdr))
    out = {}
    for X_ in THRESHOLDS:
        ys, keepmask = [], []
        for r in rows:
            t = tilts[r["ep"]]
            over = np.where(t.max(axis=1) > X_)[0]
            fs = int(over[0]) if len(over) else None
            lo, hi = r["s"] + NH, r["s"] + span - 1
            if fs is not None and fs < lo:
                keepmask.append(False); ys.append(0); continue
            keepmask.append(True)
            ys.append(1 if (fs is not None and lo <= fs <= hi) else 0)
        ys = np.array(ys); km = np.array(keepmask)
        cells = []
        for mname in METS:
            v = np.array([r[mname] for r in rows])
            cells.append(auc(v[km & (ys == 1)], v[km & (ys == 0)]))
        npos = int((km & (ys == 1)).sum()); nneg = int((km & (ys == 0)).sum())
        out[str(X_)] = {"n_pos": npos, "n_neg": nneg,
                        **{m: c for m, c in zip(METS, cells)}}
        print(f"{X_:>10.0f}{npos:>7}{nneg:>7}{100*npos/max(npos+nneg,1):>7.1f}%"
              + "".join(f"{c:>14.3f}" for c in cells))

    print("\n  AUC holding up at low thresholds -> the monitor reads a continuum the binary")
    print("  topple label discards. Collapsing toward 0.5 -> it only sees actual falls.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_chunks": len(rows), "sweep": out}, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
