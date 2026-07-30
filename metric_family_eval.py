"""
Three questions in one pass, since all three reuse the same N=50 rollouts:

(5) ftle_variance -- benchmarked against d_end/ftle_pooled/probe for the first time.
    CLAUDE.md describes it as implemented but never benchmarked: spread of the N-1 PERTURBED
    final latents around their own centroid, without referencing the original trajectory at
    all. Unlike ftle's ratio, nothing here is divided by a small early-step distance, so it
    sidesteps the whole d_start problem -- at the cost of never comparing to what the
    UNPERTURBED action would have done.

        centroid[p] = mean_j z_pert_j[T,p]
        ftle_variance[p] = mean_j (1 - cos(z_pert_j[T,p], centroid[p]))

(6) Probe calibration. R2=0.836 says the probe explains variance well, but says nothing about
    whether predicted degrees ARE actual degrees, especially near the 13-47 deg band where
    halt decisions are made. Binned reliability: mean predicted vs mean actual tilt per bin,
    plus a global regression slope/intercept (1.0 / 0.0 = perfectly calibrated).

(7) Threshold stability. Every threshold so far is a single point estimate: the p90/p95/p99
    of one held-out safe pool. Bootstrapping that pool (resampled by EPISODE, 2000 reps)
    shows how much the threshold VALUE itself would move on a different calibration sample --
    a deployment-relevant question distinct from AUC stability.

All divergence metrics use the validated low-norm mask (k=30) and p90 reduction. Labels use
the 45 deg topple definition (kept the same per section 7.22 -- redefining it would not
change anything in 20-90 deg and would be circular below it).
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
TOPPLE = 45.0
LOWNORM_K = 30
FLOOR = 1e-3


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
    ap.add_argument("--out", default="outputs/metric_family_eval.json")
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

        print(f"scoring pre-failure chunks of {len(held)} held-out episodes (N={N})", flush=True)
        rows = []
        for i, ep in enumerate(held):
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                y = 1 if (fstep is not None and lo <= fstep <= hi) else 0
                zv = roll(ep, s, keys, acts, props, N)
                if zv is None:
                    break
                zo, zn = zv[0:1], zv[1:]
                ds = ((1 - cs(zn[:, NH], zo[:, NH], dim=-1)) + 1e-4).cpu().numpy()
                de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).cpu().numpy()
                # ftle_variance: spread of the PERTURBED cloud around its own centroid,
                # never touching the original trajectory
                centroid = zn[:, -1].mean(0, keepdim=True)
                fv = (1 - cs(zn[:, -1], centroid, dim=-1)).cpu().numpy()
                nrm = zo[0, NH - 1].norm(dim=-1).cpu().numpy()
                m = keep.copy(); m[np.argsort(nrm)[:LOWNORM_K]] = False
                lp = np.where(de > FLOOR, (1.0 / NP) * np.log(de / np.median(ds[:, m])), np.nan)
                with np.errstate(invalid="ignore"):
                    lp_p = np.nanmean(lp, axis=0)
                probe_pred = float(np.c_[zv[0, -1][keep].mean(0).cpu().numpy()[None],
                                         np.ones((1, 1))] @ w)
                rows.append({
                    "ep": ep, "s": s, "y": y,
                    "probe": probe_pred,
                    "d_end": float(np.percentile(de[:, m].mean(0), 90)),
                    "ftle_pooled": float(np.nanpercentile(lp_p[m], 90)),
                    "ftle_variance": float(np.percentile(fv[:, m].mean(0), 90)),
                    "tilt_now": float(tilt[s + NH - 1].max()),
                    "tilt_fut": float(tilt[min(s + span - 1, len(tilt) - 1)].max()),
                })
            if i % 10 == 0:
                print(f"  [{i}/{len(held)}] chunks={len(rows)}", flush=True)
    env.close()

    METS = ["probe", "d_end", "ftle_pooled", "ftle_variance"]
    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")

    # ---------------- (5) ftle_variance benchmark ----------------
    print("\n=== (5) AUC, all four metrics on identical chunks ===")
    for m in METS:
        v = np.array([r[m] for r in rows])
        print(f"  {m:<14}{auc(v[ys==1], v[ys==0]):.3f}")
    print("  reference: probe 0.941, d_end 0.887, ftle_pooled 0.785 (section 7.16/7.22)")

    # ---------------- (6) probe calibration ----------------
    print("\n=== (6) probe calibration: predicted vs actual future tilt (all chunks) ===")
    pred = np.array([r["probe"] for r in rows])
    actual = np.array([r["tilt_fut"] for r in rows])
    bins = [0, 5, 10, 15, 20, 30, 45, 70, 100]
    print(f"{'pred bin':>12}{'n':>6}{'mean pred':>11}{'mean actual':>13}{'bias':>8}")
    print("-" * 50)
    cal_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (pred >= lo) & (pred < hi)
        if sel.sum() < 3:
            continue
        mp, ma = pred[sel].mean(), actual[sel].mean()
        cal_rows.append({"lo": lo, "hi": hi, "n": int(sel.sum()), "mean_pred": float(mp),
                         "mean_actual": float(ma), "bias": float(ma - mp)})
        print(f"{f'{lo}-{hi}':>12}{int(sel.sum()):>6}{mp:>11.2f}{ma:>13.2f}{ma-mp:>8.2f}")
    slope, intercept = np.polyfit(pred, actual, 1)
    print(f"\n  global regression: actual = {slope:.3f} * pred + {intercept:.2f}")
    print("  well-calibrated => slope near 1.0, intercept near 0, bias near 0 in every bin")
    print("  most decision-relevant band: pred 8-47 deg (between p80 and p99 thresholds)")

    # ---------------- (7) threshold stability ----------------
    print("\n=== (7) threshold VALUE stability under resampling (2000 reps, by episode) ===")
    safe = [r for r in rows if r["y"] == 0]
    eps_u = sorted({r["ep"] for r in safe})
    idx = {e: [i for i, r in enumerate(safe) if r["ep"] == e] for e in eps_u}
    rng = np.random.default_rng(0)
    print(f"  {len(safe)} safe chunks across {len(eps_u)} episodes")
    thr_stats = {}
    for m in METS:
        v = np.array([r[m] for r in safe])
        obs = {q: float(np.nanpercentile(v, q)) for q in (80, 90, 95, 99)}
        boot = {q: [] for q in (80, 90, 95, 99)}
        for _ in range(2000):
            sel = np.concatenate([idx[e] for e in rng.choice(eps_u, len(eps_u), replace=True)])
            vv = v[sel]
            for q in (80, 90, 95, 99):
                boot[q].append(np.nanpercentile(vv, q))
        print(f"\n  {m}")
        print(f"  {'q':>5}{'point est':>12}{'boot mean':>12}{'boot std':>11}{'CV':>8}"
              f"{'95% CI':>22}")
        thr_stats[m] = {}
        for q in (80, 90, 95, 99):
            b = np.array(boot[q]); lo, hi = np.percentile(b, [2.5, 97.5])
            cv = b.std() / max(abs(b.mean()), 1e-9)
            thr_stats[m][q] = {"point": obs[q], "boot_mean": float(b.mean()),
                               "boot_std": float(b.std()), "cv": float(cv),
                               "ci": [float(lo), float(hi)]}
            print(f"  p{q:<4}{obs[q]:>12.4f}{b.mean():>12.4f}{b.std():>11.4f}{cv:>8.2f}"
                  f"{f'[{lo:.4f}, {hi:.4f}]':>22}")
    print("\n  large CV / wide CI => a single calibration run picks an unstable threshold;")
    print("  tight CI => the threshold is reproducible across different calibration samples")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "calibration": cal_rows,
               "calibration_fit": {"slope": float(slope), "intercept": float(intercept)},
               "threshold_stability": thr_stats,
               "rows": rows},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
