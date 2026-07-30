"""
Is the N=50 max a good estimate of maximum expansion -- and is the response nonlinear?

Two questions, one rollout sweep, both bearing on whether to replace the sampled estimator
with an exact Jacobian sigma_max.

(1) SATURATION. The action perturbation lives in 8 steps x 3 dims = 24 dimensions. A random
    direction there has expected alignment ~1/sqrt(24) = 0.20 with the top singular vector;
    the best of 50 draws maybe 0.5-0.6. So `max_j d_end` over N=50 is a Monte Carlo LOWER
    BOUND on the true maximum expansion, and how tight it is varies chunk to chunk -- which
    is variance injected straight into the score.

    Sweeping N and watching max_j d_end: if it plateaus by 50, random sampling already finds
    the dominant direction and an exact Jacobian buys precision only. If it keeps climbing,
    expansion is being left on the table and sigma_max should be both larger and steadier.

(2) NONLINEARITY. A Jacobian is a linearization: it measures infinitesimal divergence. But
    sigma=0.05 exceeds one action std (0.032), and toppling is a basin-boundary crossing,
    not smooth expansion. If finite perturbations are crossing that boundary, the per-
    perturbation d_end distribution on UNSAFE chunks should be BIMODAL -- some perturbations
    topple the block, some do not. Unimodal means the response is smooth over this range and
    a linearization loses little.

    Measured by: bimodality coefficient BC = (skew^2 + 1) / kurtosis, where BC > 0.555
    indicates bimodality (uniform reference), plus the normalised gap between the two halves
    of a 2-means split. Reported separately for safe and unsafe chunks -- the contrast is the
    point, since only unsafe chunks sit near a boundary.

Rollouts run in batches so N=400 fits in 8 GB.
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
NS = [10, 25, 50, 100, 200, 400]


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def bimodality(x):
    """BC = (skew^2 + 1)/kurtosis; > 0.555 suggests bimodal. Also a 2-means gap."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 8 or x.std() < 1e-12:
        return np.nan, np.nan
    z = (x - x.mean()) / x.std()
    sk = (z ** 3).mean(); ku = (z ** 4).mean()
    bc = (sk ** 2 + 1) / max(ku, 1e-9)
    # crude 2-means on a sorted split, maximising between-group separation
    xs = np.sort(x); best = 0.0
    for i in range(2, len(xs) - 2):
        a, b = xs[:i], xs[i:]
        sep = (b.mean() - a.mean()) / (x.std() + 1e-12)
        best = max(best, sep)
    return float(bc), float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=400)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-safe", type=int, default=100)
    ap.add_argument("--out", default="outputs/perturbation_coverage.json")
    args = ap.parse_args()

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
        print(f"{len(targets)} chunks ({len(pos)} unsafe, {len(neg)} safe), N up to {args.n_max}",
              flush=True)

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
            a0 = torch.from_numpy(acts[s:s + span]).float().to(dev)
            pro = ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd)

            # nominal rollout once
            with torch.no_grad():
                zc, _ = model.rollout({"visual": vis[:NH].unsqueeze(0),
                                       "proprio": pro.unsqueeze(0)},
                                      ((a0.unsqueeze(0) - am) / asd))
            z_ref = zc["visual"][0, -1]

            g = torch.Generator(device=dev); g.manual_seed(s)
            per_pert = []
            done = 0
            while done < args.n_max:
                b = min(args.batch, args.n_max - done)
                a = a0.unsqueeze(0).repeat(b, 1, 1)
                a[:, :, :3] += torch.randn(b, span, 3, device=dev, generator=g) * args.noise_std
                obs = {"visual": vis[:NH].unsqueeze(0).repeat(b, 1, 1, 1, 1),
                       "proprio": pro.unsqueeze(0).repeat(b, 1, 1)}
                with torch.no_grad():
                    z, _ = model.rollout(obs, (a - am) / asd)
                d = ((1 - cs(z["visual"][:, -1], z_ref.unsqueeze(0), dim=-1)) + 1e-4)
                # one scalar per perturbation: p90 over kept patches
                per_pert.append(np.percentile(d[:, keep].cpu().numpy(), 90, axis=1))
                done += b
            rows.append({"ep": ep, "y": y, "per_pert": np.concatenate(per_pert)})
            if i % 20 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")

    # ---------- (1) saturation ----------
    print("\n=== SATURATION: does max_j d_end keep growing with N? ===")
    print("  (mean over chunks of the max, and of max_N / max_400)")
    print(f"{'N':>6}{'max (unsafe)':>14}{'max (safe)':>12}{'frac of N=400':>15}")
    print("-" * 47)
    ref = np.array([r["per_pert"][:args.n_max].max() for r in rows])
    for n in NS:
        if n > args.n_max:
            continue
        mx = np.array([r["per_pert"][:n].max() for r in rows])
        print(f"{n:>6}{mx[ys==1].mean():>14.4f}{mx[ys==0].mean():>12.4f}"
              f"{(mx/ref).mean():>15.3f}")
    print("  plateau => random sampling already finds the dominant direction;")
    print("  still climbing => an exact Jacobian sigma_max would be larger and steadier")

    # how noisy is the N=50 max? resample disjoint blocks of 50
    if args.n_max >= 400:
        cv = []
        for r in rows:
            blocks = [r["per_pert"][k*50:(k+1)*50].max() for k in range(args.n_max // 50)]
            cv.append(np.std(blocks) / max(np.mean(blocks), 1e-9))
        print(f"\n  chunk-to-chunk CV of the N=50 max across disjoint blocks: {np.mean(cv):.4f}")
        print("  (this is the estimator noise an exact Jacobian would remove entirely)")

    # ---------- (2) nonlinearity ----------
    print("\n=== NONLINEARITY: is the per-perturbation distribution bimodal? ===")
    print("  bimodal on unsafe chunks => finite perturbations cross a basin boundary,")
    print("  which a linearisation cannot see. unimodal => linearisation loses little.")
    print(f"{'group':>10}{'BC (>0.555 bimodal)':>22}{'2-means gap':>14}{'n':>6}")
    print("-" * 52)
    out = {}
    for name, sel in (("unsafe", ys == 1), ("safe", ys == 0)):
        bcs, gaps = [], []
        for r, keepit in zip(rows, sel):
            if not keepit:
                continue
            bc, gap = bimodality(r["per_pert"])
            if np.isfinite(bc):
                bcs.append(bc); gaps.append(gap)
        print(f"{name:>10}{np.mean(bcs):>22.3f}{np.mean(gaps):>14.3f}{len(bcs):>6}")
        out[name] = {"bc": float(np.mean(bcs)), "gap": float(np.mean(gaps))}
        frac = np.mean([b > 0.555 for b in bcs])
        print(f"{'':>10}fraction of chunks with BC > 0.555: {frac:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_chunks": len(rows), "bimodality": out,
               "saturation": {str(n): float(np.array([r["per_pert"][:n].max()
                                                      for r in rows]).mean())
                              for n in NS if n <= args.n_max}},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
