"""
Are PCA truncation and the low-norm mask the same win twice, or two wins?

Both landed at essentially the same place on the same subset:

    PCA truncation to top-4 PCs      p90 AUC 0.855   (from 0.673 at full 384)
    drop the 30 lowest-||z|| patches p90 AUC 0.854   (from 0.759 on the full set)

They could easily be redundant. PC1 correlates with foreground/background, and low-||z||
patches are mostly background, so both may be deleting the same blank table. If redundant,
pick the simpler one and stop; if additive, stacking should clear 0.85. Only a joint sweep
over (m, k) on identical rollouts can tell them apart.

This also reruns the PC1 foreground mask correctly. The earlier attempt thresholded on
|PC1|, but PC1 separates foreground from background BY SIGN -- taking the absolute value
keeps both extremes and discards the middle, which is a "distance from average" mask, not
a foreground mask. Here both signed directions are tried, and which one is foreground is
determined from the data (by checking which side has the higher mean patch norm) rather
than assumed.
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
    ap.add_argument("--out", default="outputs/pca_mask_combo.json")
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
            rows.append({
                "ep": ep, "y": y,
                "z_end": zv[0, -1].cpu().numpy().astype(np.float32),
                "zp_end": zv[1:, -1].cpu().numpy().astype(np.float32),
                "z_obs": zv[0, NH - 1].cpu().numpy().astype(np.float32),
                "norm": zv[0, NH - 1].norm(dim=-1).cpu().numpy(),
            })
            if i % 50 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")

    bank = np.concatenate([r["z_end"][keep] for r, y in zip(rows, ys) if y == 0], 0)
    mu = bank.mean(0)
    _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    print(f"PCA on {bank.shape[0]} safe patch vectors; top1={ev[0]:.3f} top4={ev[:4].sum():.3f}")

    def d_end(r, m):
        if m >= 384:
            c = torch.nn.functional.cosine_similarity(
                torch.from_numpy(r["zp_end"]), torch.from_numpy(r["z_end"])[None], dim=-1).numpy()
            return 1 - c
        P = Vt[:m]
        o = (r["z_end"] - mu) @ P.T
        p = (r["zp_end"] - mu) @ P.T
        num = (p * o[None]).sum(-1)
        den = np.linalg.norm(p, axis=-1) * np.linalg.norm(o, axis=-1)[None] + 1e-9
        return 1 - num / den

    def score(d, mask, red):
        v = d[:, mask]
        v = v[np.isfinite(v)]
        if v.size < 4:
            return np.nan
        return {"mean": v.mean(), "p90": np.percentile(v, 90), "max": v.max()}[red]

    # ---------- joint sweep: PCA truncation x low-norm mask ----------
    print("\n=== PCA truncation (m) x low-norm mask (k), p90 AUC ===")
    print("  redundant => rows flat across k;  additive => best cell beats both margins")
    KS = [0, 10, 20, 30]
    MS = [4, 8, 16, 384]
    cache = {m: [d_end(r, m) for r in rows] for m in MS}
    hdr = "m / k"
    print(f"{hdr:<8}" + "".join(f"{('k='+str(k)):>10}" for k in KS))
    print("-" * (8 + 10 * len(KS)))
    best = ("", -1)
    for m in MS:
        cells = []
        for k in KS:
            v = []
            for r, d in zip(rows, cache[m]):
                msk = keep.copy()
                if k:
                    msk[np.argsort(r["norm"])[:k]] = False
                v.append(score(d, msk, "p90"))
            v = np.array(v); a = auc(v[ys == 1], v[ys == 0]); cells.append(a)
            if a > best[1]:
                best = (f"m={m}, k={k}", a)
        print(f"{('m='+str(m)):<8}" + "".join(f"{c:>10.3f}" for c in cells))
    print(f"\nbest cell: {best[0]}  AUC {best[1]:.3f}")
    print("margins: m=4/k=0 and m=384/k=30 are the two single-fix baselines")

    # ---------- signed PC1 foreground mask, done properly ----------
    pc1_all = np.concatenate([((r["z_obs"] - mu) @ Vt[0])[keep] for r in rows])
    nrm_all = np.concatenate([r["norm"][keep] for r in rows])
    hi_side = nrm_all[pc1_all > np.median(pc1_all)].mean()
    lo_side = nrm_all[pc1_all <= np.median(pc1_all)].mean()
    fg_sign = +1 if hi_side > lo_side else -1
    print(f"\n=== signed PC1 mask ===")
    print(f"  mean ||z||: PC1-high {hi_side:.1f} vs PC1-low {lo_side:.1f}"
          f"  -> foreground is PC1 {'POSITIVE' if fg_sign > 0 else 'NEGATIVE'}")
    print(f"{'keep %':>8}{'kept':>7}{'mean':>9}{'p90':>9}{'max':>9}")
    print("-" * 42)
    for pct in (100, 75, 50, 40, 25):
        cells, kept = [], []
        for red in ("mean", "p90", "max"):
            v = []
            for r, d in zip(rows, cache[384]):
                pc1 = ((r["z_obs"] - mu) @ Vt[0]) * fg_sign
                msk = keep.copy()
                if pct < 100:
                    msk &= pc1 >= np.percentile(pc1[keep], 100 - pct)
                if msk.sum() < 4:
                    v.append(np.nan); continue
                kept.append(msk.sum())
                v.append(score(d, msk, red))
            v = np.array(v); cells.append(auc(v[ys == 1], v[ys == 0]))
        print(f"{pct:>8}{int(np.mean(kept)) if kept else 0:>7}"
              + "".join(f"{c:>9.3f}" for c in cells))
    print("  compare: |PC1| version gave p90 0.799/0.793/0.785/0.780/0.739 (100..25%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "best": best[0], "best_auc": best[1],
               "fg_sign": int(fg_sign)}, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
