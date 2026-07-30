"""
The apples-to-apples comparison: probe vs divergence, on the SAFETY task.

The probe's headline numbers (R^2 0.836, AUC 0.992) and the monitor's operating points
(recall .52, precision .129) are not comparable -- different target (tilt angle vs safety
label), different data (jenga_tilt_100 vs jenga_noise_50), and crucially the probe was
scored on ALL chunks including post-topple ones, where "still fallen 8 steps from now" is
trivially predictable. Quoting 0.992 beside 0.206 implies something neither number supports.

This scores both on identical chunks of identical data, under the monitor's own rules:

  * chunk is positive iff the failure step falls in its 8-step prediction horizon
  * chunks strictly after the failure are DROPPED (no credit for detecting the aftermath)
  * thresholds are percentiles of the SAFE score distribution only
  * the probe is fit on one half of the episodes and scored on the other

Labels come from the LMDB's own `_tilt` array rather than a separate labels.json: tilt is
already aligned to the LMDB indices by construction, so failure_step = first index where
either adjacent block exceeds 45 deg. That sidesteps the timestamp-realignment step that
labels.json needs, and it cannot drift out of sync with the frames.

Three scores per chunk, all from one N=50 rollout:
    probe     linear readout of the predicted latent -> tilt at t+8
    d_end     mean over perturbations of final-step cosine divergence (p90 over patches)
    nominal   how much the unperturbed predicted scene changes (p90 over patches)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/sanger/wksp/panda_express/tasks/jenga_tilt_100/jenga_tilt.lmdb")
    ap.add_argument("--n-fit-episodes", type=int, default=50)
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/probe_vs_divergence.json")
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

    def roll(txn, ep, s, keys, acts, props, n_pert):
        raw = [txn.get(keys[s + j].encode()) for j in range(span)]
        if any(r is None for r in raw):
            return None
        vis = tf(torch.from_numpy(np.stack([np.transpose(dec(r), (2, 0, 1)) for r in raw])
                                  ).float().to(dev) / 255.)
        a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(n_pert, 1, 1)
        if n_pert > 1:
            g = torch.Generator(device=dev); g.manual_seed(s)
            a[1:, :, :3] += torch.randn(n_pert - 1, span, 3, device=dev,
                                        generator=g) * args.noise_std
        obs = {"visual": vis[:NH].unsqueeze(0).repeat(n_pert, 1, 1, 1, 1),
               "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                           ).unsqueeze(0).repeat(n_pert, 1, 1)}
        with torch.no_grad():
            z, _ = model.rollout(obs, (a - am) / asd)
        return z["visual"]

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = sorted(meta["episodes"], key=lambda e: int(e))
        fit_eps, held_eps = eps[:args.n_fit_episodes], eps[args.n_fit_episodes:]

        def load(ep):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            return keys, acts, props, tilt, min(len(keys), len(acts), len(props), len(tilt))

        # ---------- fit the probe on the fit half (N=1) ----------
        print(f"fitting probe on {len(fit_eps)} episodes (N=1)", flush=True)
        X, Y = [], []
        for i, ep in enumerate(fit_eps):
            keys, acts, props, tilt, n = load(ep)
            for s in range(0, n - span, NP):
                zv = roll(txn, ep, s, keys, acts, props, 1)
                if zv is None:
                    break
                X.append(zv[0, -1][keep].mean(0).cpu().numpy())
                Y.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
            if i % 10 == 0:
                print(f"  [{i}/{len(fit_eps)}] n={len(X)}", flush=True)
        X = np.stack(X); Y = np.array(Y)
        w = ridge_fit(X, Y, args.lam)
        print(f"  fit on {len(X)} chunks, train R2 "
              f"{1 - ((Y - ridge_apply(X, w))**2).sum() / ((Y - Y.mean())**2).sum():.3f}")

        # ---------- score the held-out half under the MONITOR's rules (N=50) ----------
        print(f"\nscoring {len(held_eps)} held-out episodes (N={N})", flush=True)
        rows = []
        for i, ep in enumerate(held_eps):
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break                      # post-failure: aftermath, not prediction
                y = 1 if (fstep is not None and lo <= fstep <= hi) else 0
                zv = roll(txn, ep, s, keys, acts, props, N)
                if zv is None:
                    break
                zo, zn = zv[0:1], zv[1:]
                de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).mean(0).cpu().numpy()
                nm = (1 - cs(zo[0, NH], zo[0, -1], dim=-1)).cpu().numpy()
                nrm = zo[0, NH - 1].norm(dim=-1).cpu().numpy()
                m = keep.copy(); m[np.argsort(nrm)[:LOWNORM_K]] = False
                rows.append({
                    "ep": ep, "y": y,
                    "probe": float(ridge_apply(zv[0, -1][keep].mean(0).cpu().numpy()[None], w)[0]),
                    "d_end": float(np.percentile(de[m], 90)),
                    "nominal": float(np.percentile(nm[m], 90)),
                    "gt_tilt": float(tilt[min(s + span - 1, len(tilt) - 1)].max()),
                })
            if i % 10 == 0:
                print(f"  [{i}/{len(held_eps)}] chunks={len(rows)}", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} held-out chunks | {int(ys.sum())} unsafe "
          f"({100*ys.mean():.1f}%) | {int((1-ys).sum())} safe")
    print(f"trivial 'always safe' accuracy = {1-ys.mean():.4f}")

    METHODS = [("probe (tilt at t+8)", "probe"),
               ("d_end p90 / k=30", "d_end"),
               ("nominal p90 / k=30", "nominal")]
    results = {}
    for name, key in METHODS:
        v = np.array([r[key] for r in rows])
        a = auc(v[ys == 1], v[ys == 0])
        results[name] = {"auc": a, "points": {}}
        print(f"\n=== {name}   AUC {a:.3f} ===")
        print(f"{'thr@':>6}{'value':>10}{'TP':>5}{'FP':>6}{'FN':>5}"
              f"{'recall':>8}{'prec':>7}{'acc':>8}{'F1':>7}{'FP/ep':>8}")
        for q in (75, 80, 85, 90, 95, 99):
            t = np.percentile(v[ys == 0], q); p = v > t
            tp = int((p & (ys == 1)).sum()); fp = int((p & (ys == 0)).sum())
            fn = int((~p & (ys == 1)).sum()); tn = int((~p & (ys == 0)).sum())
            rec = tp / max(tp + fn, 1); pre = tp / max(tp + fp, 1)
            acc = (tp + tn) / len(ys); f1 = 2 * rec * pre / max(rec + pre, 1e-9)
            print(f"{'p'+str(q):>6}{t:>10.4f}{tp:>5}{fp:>6}{fn:>5}"
                  f"{rec:>8.3f}{pre:>7.3f}{acc:>8.4f}{f1:>7.3f}{fp/len(set(r['ep'] for r in rows)):>8.2f}")
            results[name]["points"][f"p{q}"] = {"recall": rec, "precision": pre,
                                                "accuracy": acc, "f1": f1}

    # paired bootstrap: does the probe actually beat the divergence metrics?
    eps_u = sorted({r["ep"] for r in rows})
    idx = {e: [i for i, r in enumerate(rows) if r["ep"] == e] for e in eps_u}
    rng = np.random.default_rng(0)
    print("\n=== paired cluster bootstrap over episodes, 2000 reps ===")
    print(f"{'comparison':<40}{'A':>7}{'B':>7}{'A-B':>8}{'95% CI':>18}{'P(A>B)':>9}")
    print("-" * 89)
    for bn, bk in [("d_end p90 / k=30", "d_end"), ("nominal p90 / k=30", "nominal")]:
        va = np.array([r["probe"] for r in rows]); vb = np.array([r[bk] for r in rows])
        a0 = auc(va[ys == 1], va[ys == 0]); b0 = auc(vb[ys == 1], vb[ys == 0]); ds = []
        for _ in range(2000):
            sel = np.concatenate([idx[e] for e in rng.choice(eps_u, len(eps_u), replace=True)])
            yy = ys[sel]
            if yy.sum() == 0 or (1 - yy).sum() == 0:
                continue
            ds.append(auc(va[sel][yy == 1], va[sel][yy == 0])
                      - auc(vb[sel][yy == 1], vb[sel][yy == 0]))
        ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"{'probe  vs  ' + bn:<40}{a0:>7.3f}{b0:>7.3f}{a0-b0:>8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{(ds>0).mean():>9.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_chunks": len(rows), "n_unsafe": int(ys.sum()), "results": results},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
