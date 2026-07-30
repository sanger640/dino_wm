"""
Can the FTLE ratio be made noise-resilient without abandoning it?

Diagnosis (from ftle_with_fixes.py): the problem is the DENOMINATOR. d_start is measured
after a single prediction step, so it is ~7x smaller than d_end (0.0047 vs 0.0290) and
noisier (CV 0.42 vs 0.28), and it enters multiplicatively. It also carries signal in the
SAME direction as d_end (AUC 0.607 alone), so dividing cancels signal as well as normalising.

Five ways to keep the exponential-growth-rate idea while removing the fragility:

  slope       Fit log d(t) against t over ALL timesteps t = NH..T by least squares, instead
              of using only the two endpoints. This IS the Lyapunov exponent by definition
              (exponential growth rate) and uses 9 measurements rather than 2, so endpoint
              noise -- especially the bad endpoint -- is averaged down rather than divided by.

  ratio_means log(mean_j d_end / mean_j d_start) instead of mean_j log(d_end/d_start).
              Mean-of-ratios is dominated by the few perturbations where the denominator is
              near zero; ratio-of-means is not.

  pooled      One denominator per chunk (median over patches) instead of a per-patch one.
              Removes per-patch denominator noise while keeping the normalisation.

  shrunk      d_end / (d_start + eps) with eps at the scale of d_start itself, rather than
              the 1e-4 floor. Interpolates between the ratio (eps=0) and d_end (eps->inf),
              so the sweep shows directly how much of the ratio is worth keeping.

  median      Median over perturbations instead of mean -- trims the blown-up tail.

All variants use the validated low-norm mask (drop the 30 lowest-||z|| patches) and are
scored on identical rollouts, with d_end as the reference line.
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
LMDB = "/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"
LABELS = "/home/sanger/wksp/panda_express/labels_noise100.json"
LOWNORM_K = 30
EPS0 = 1e-4


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-safe", type=int, default=200)
    ap.add_argument("--out", default="outputs/robust_ftle.json")
    args = ap.parse_args()
    N = args.n_perturb

    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_model(Path("outputs/model_latest_single.pth"), cfg, dev); model.eval()
    tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
    pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)
    labels = json.load(open(LABELS))
    keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
    span = NH + NP
    cs = torch.nn.functional.cosine_similarity
    rng = np.random.default_rng(0)
    env = lmdb.open(LMDB, readonly=True, lock=False)

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        pos, neg = [], []
        for ep, v in labels.items():
            if ep not in meta["episodes"]:
                continue
            keys = meta["episodes"][ep]["keys"]["cam2"]
            f = v["failure_step"] if v["outcome"] == "failure" else None
            for s in range(0, len(keys) - span, NP):
                lo, hi = s + NH, s + span - 1
                if f is not None and f < lo:
                    break
                (pos if (f is not None and f <= hi) else neg).append((ep, s))
        neg = [neg[i] for i in rng.choice(len(neg), min(args.n_safe, len(neg)), replace=False)]
        targets = [(e, s, 1) for e, s in pos] + [(e, s, 0) for e, s in neg]
        print(f"{len(targets)} chunks ({len(pos)} unsafe, {len(neg)} safe)", flush=True)

        rows = []
        for i, (ep, s, y) in enumerate(targets):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            if s + span > min(len(acts), len(props)):
                continue
            raw = [txn.get(keys[s + j].encode()) for j in range(span)]
            if any(r is None for r in raw):
                continue
            vis = tf(torch.from_numpy(np.stack([np.transpose(dec(r), (2, 0, 1)) for r in raw])
                                      ).float().to(dev) / 255.)
            g = torch.Generator(device=dev); g.manual_seed(s)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
            a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                               ).unsqueeze(0).repeat(N, 1, 1)}
            with torch.no_grad():
                z, _ = model.rollout(obs, (a - am) / asd)
            zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
            # distance at EVERY predicted timestep, not just the endpoints: (J, T, P)
            d = torch.stack([(1 - cs(zn[:, t], zo[:, t], dim=-1)) + EPS0
                             for t in range(NH, zv.shape[1])], dim=1).cpu().numpy()
            rows.append({"ep": ep, "y": y, "d": d.astype(np.float32),
                         "norm": zo[0, NH - 1].norm(dim=-1).cpu().numpy()})
            if i % 50 == 0:
                print(f"  [{i}/{len(targets)}] d shape {d.shape}", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    T = rows[0]["d"].shape[1]
    tt = np.arange(T, dtype=np.float64)
    tc = tt - tt.mean(); tvar = (tc ** 2).sum()
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe "
          f"| {T} timesteps per rollout")

    def variants(r):
        """All per-patch score maps for one chunk -> dict name -> (P,) array."""
        d = r["d"].astype(np.float64)              # (J, T, P)
        ds, de = d[:, 0], d[:, -1]                 # first predicted step, last
        L = np.log(d)
        out = {}
        # --- current two-point ratio ---
        out["ftle_2pt (current)"] = ((1.0 / NP) * (np.log(de) - np.log(ds))).mean(0)
        out["ftle_2pt_median"] = np.median((1.0 / NP) * (np.log(de) - np.log(ds)), axis=0)
        # --- regression slope over all timesteps ---
        slope = np.einsum("t,jtp->jp", tc, L) / tvar
        out["ftle_slope"] = slope.mean(0)
        out["ftle_slope_median"] = np.median(slope, axis=0)
        # perturbations averaged BEFORE the slope
        out["ftle_slope_of_mean"] = np.einsum("t,tp->p", tc, np.log(d.mean(0))) / tvar
        # --- ratio of means instead of mean of ratios ---
        out["ftle_ratio_of_means"] = (1.0 / NP) * (np.log(de.mean(0)) - np.log(ds.mean(0)))
        # --- pooled denominator: one d_start per chunk ---
        pooled = np.median(ds)
        out["ftle_pooled_den"] = ((1.0 / NP) * (np.log(de) - np.log(pooled))).mean(0)
        # --- shrunk denominator ---
        med = np.median(ds)
        for mult, tag in ((0.5, "0.5x"), (1.0, "1x"), (4.0, "4x")):
            eps = mult * med
            out[f"ftle_shrunk_{tag}"] = ((1.0 / NP) * (np.log(de) - np.log(ds + eps))).mean(0)
        # --- reference ---
        out["d_end (reference)"] = de.mean(0)
        return out

    names = list(variants(rows[0]).keys())
    print(f"\n=== AUC, low-norm mask k={LOWNORM_K}, identical rollouts ===")
    hdr = f"{'variant':<24}" + "".join(f"{r:>9}" for r in ("mean", "p90", "max"))
    print(hdr); print("-" * len(hdr))
    scores = {n: {} for n in names}
    per_chunk = {n: [] for n in names}
    for r in rows:
        v = variants(r)
        m = keep.copy(); m[np.argsort(r["norm"])[:LOWNORM_K]] = False
        for n in names:
            x = v[n][m]; x = x[np.isfinite(x)]
            per_chunk[n].append((x.mean(), np.percentile(x, 90), x.max())
                                if x.size >= 4 else (np.nan,) * 3)
    best = ("", -1)
    for n in names:
        arr = np.array(per_chunk[n])
        cells = [auc(arr[ys == 1, c], arr[ys == 0, c]) for c in range(3)]
        for c, red in enumerate(("mean", "p90", "max")):
            scores[n][red] = cells[c]
            if cells[c] > best[1] and "reference" not in n:
                best = (f"{n} / {red}", cells[c])
        print(f"{n:<24}" + "".join(f"{c:>9.3f}" for c in cells))
    ref = max(scores["d_end (reference)"].values())
    print(f"\nbest FTLE variant : {best[0]}  AUC {best[1]:.3f}")
    print(f"d_end reference   : {ref:.3f}")
    print(f"gap               : {best[1] - ref:+.3f}")
    print("  current two-point ratio scored 0.710 here previously; 0.599 unmasked originally")

    # paired bootstrap of the best variant against d_end
    eps_u = sorted({r["ep"] for r in rows})
    idx = {e: [i for i, r in enumerate(rows) if r["ep"] == e] for e in eps_u}
    bn, br = best[0].rsplit(" / ", 1)
    ci = {"mean": 0, "p90": 1, "max": 2}[br]
    va = np.array(per_chunk[bn])[:, ci]
    rc = max(scores["d_end (reference)"], key=lambda k: scores["d_end (reference)"][k])
    vb = np.array(per_chunk["d_end (reference)"])[:, {"mean": 0, "p90": 1, "max": 2}[rc]]
    rng2 = np.random.default_rng(0); ds_ = []
    for _ in range(2000):
        sel = np.concatenate([idx[e] for e in rng2.choice(eps_u, len(eps_u), replace=True)])
        yy = ys[sel]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        ds_.append(auc(va[sel][yy == 1], va[sel][yy == 0])
                   - auc(vb[sel][yy == 1], vb[sel][yy == 0]))
    ds_ = np.array(ds_); lo, hi = np.percentile(ds_, [2.5, 97.5])
    print(f"\n{bn}/{br} vs d_end/{rc}: {best[1]:.3f} vs {ref:.3f}"
          f"  diff {best[1]-ref:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(A>B)={(ds_>0).mean():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "auc": scores, "best": best, "d_end_ref": ref},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
