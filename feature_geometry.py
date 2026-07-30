"""
Is a cosine over all 384 dimensions the right readout, and what do those dimensions hold?

Measured earlier: whitening the feature space (upweighting low-variance dimensions) HURT
badly -- patch AUC 0.960 -> 0.869, chunk 0.780 -> 0.698 -- despite a real 7.3x anisotropy.
That says signal concentrates in the high-variance directions. PCA truncation is the exact
opposite operation, so it should HELP. This tests that prediction, plus two others:

  PCA truncation   cosine restricted to the top-m principal components. If the tail is
                   nuisance variation (lighting, shadow, predictor error), deleting it
                   should raise AUC. If AUC is flat in m, the signal is spread across the
                   whole spectrum and no linear subspace helps.

  PC1 foreground   DINOv2's first principal component over patch tokens is known to
                   separate foreground from background. Used as an automatic task mask,
                   this is the principled version of the image-occupancy mask that failed
                   (that one keyed on the checkerboard's texture; this keys on features).
                   Expected to behave like the low-norm mask, since low-norm and background
                   patches largely coincide.

The linear tilt probe that would sit alongside these is deliberately NOT here: sim.py's
reset randomises block pose without saving a seed, so re-simulating produces a different
rollout than the stored frames, and the trajectory JSONs keep only episode-level
peak_tilt_deg. That probe needs a re-record with per-frame block state first.

Both tests fit their basis on SAFE chunks only, so the calibration stays zero-shot.
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
NH, NP, GRID = 3, 8, 14
dev = "cuda"
LMDB = "/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"
LABELS = "/home/sanger/wksp/panda_express/labels_noise100.json"
MS = [4, 8, 16, 32, 64, 128, 384]


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
    ap.add_argument("--out", default="outputs/feature_geometry.json")
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
    env = lmdb.open(LMDB, readonly=True, lock=False)
    keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
    span = NH + NP
    cs = torch.nn.functional.cosine_similarity
    rng = np.random.default_rng(0)

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
        print(f"{len(targets)} chunks ({len(pos)} unsafe, {len(neg)} safe)")

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
            imgs = [dec(r) for r in raw]
            vis = tf(torch.from_numpy(np.stack([np.transpose(im, (2, 0, 1)) for im in imgs])
                                      ).float().to(dev) / 255.)
            g = torch.Generator(device=dev); g.manual_seed(s)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
            a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                               ).unsqueeze(0).repeat(N, 1, 1)}
            with torch.no_grad():
                z, _ = model.rollout(obs, (a - am) / asd)
            zv = z["visual"]
            rows.append({
                "ep": ep, "y": y,
                "z_obs": zv[0, NH - 1].cpu().numpy().astype(np.float32),   # real encoded frame
                "z_end": zv[0, -1].cpu().numpy().astype(np.float32),       # predicted final
                "zp_end": zv[1:, -1].cpu().numpy().astype(np.float32),     # perturbed finals
            })
            if i % 25 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")

    # ---------- PCA basis from SAFE chunks only (keeps the calibration zero-shot) ----------
    bank = np.concatenate([r["z_end"][keep] for r, yy in zip(rows, ys) if yy == 0], 0)
    mu = bank.mean(0)
    U, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    print(f"\nPCA on {bank.shape[0]} safe patch vectors")
    print("  variance explained: " + "  ".join(
        f"top{m}={ev[:m].sum():.3f}" for m in (1, 4, 16, 64, 128)))

    def d_end_in(r, m):
        """cosine between perturbed and original finals, inside the top-m PC subspace."""
        P = Vt[:m]
        o = (r["z_end"] - mu) @ P.T
        p = (r["zp_end"] - mu) @ P.T
        num = (p * o[None]).sum(-1)
        den = np.linalg.norm(p, axis=-1) * np.linalg.norm(o, axis=-1)[None] + 1e-9
        return 1 - num / den

    print(f"\n=== PCA TRUNCATION: cosine inside top-m PCs ===")
    print("  prediction: AUC should RISE as m shrinks, since whitening (the opposite) hurt")
    print(f"{'m':>6}{'var':>8}{'mean':>9}{'p90':>9}{'max':>9}")
    print("-" * 41)
    for m in MS:
        de = [d_end_in(r, m) for r in rows]
        cells = []
        for red in ("mean", "p90", "max"):
            v = np.array([{"mean": d[:, keep].mean(),
                           "p90": np.percentile(d[:, keep], 90),
                           "max": d[:, keep].max()}[red] for d in de])
            cells.append(auc(v[ys == 1], v[ys == 0]))
        print(f"{m:>6}{ev[:m].sum():>8.3f}" + "".join(f"{c:>9.3f}" for c in cells))

    # ---------- PC1 as a foreground mask ----------
    print("\n=== PC1 FOREGROUND MASK (drop patches by |PC1| percentile) ===")
    print(f"{'keep pct':>9}{'kept':>7}{'mean':>9}{'p90':>9}{'max':>9}")
    print("-" * 43)
    base = [1 - cs(torch.from_numpy(r["zp_end"]), torch.from_numpy(r["z_end"])[None],
                   dim=-1).numpy() for r in rows]
    for pct in (100, 75, 50, 40, 25):
        cells, kept = [], []
        for red in ("mean", "p90", "max"):
            v = []
            for r, d in zip(rows, base):
                pc1 = np.abs((r["z_obs"] - mu) @ Vt[0])
                m = keep.copy()
                if pct < 100:
                    m &= pc1 >= np.percentile(pc1[keep], 100 - pct)
                if m.sum() < 4:
                    v.append(np.nan); continue
                kept.append(m.sum())
                x = d[:, m]
                v.append({"mean": x.mean(), "p90": np.percentile(x, 90), "max": x.max()}[red])
            v = np.array(v); cells.append(auc(v[ys == 1], v[ys == 0]))
        print(f"{pct:>9}{int(np.mean(kept)):>7}" + "".join(f"{c:>9.3f}" for c in cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "evr": ev[:128].tolist()}, open(args.out, "w"))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
