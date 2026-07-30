"""
Does the FTLE ratio become competitive once the fixes are applied?

Every measurement of the ratio so far used the ORIGINAL setup: AUC 0.599 with the double max,
0.685 for the better reduction, against 0.799-0.854 for d_end. The conclusion "drop the
denominator" was drawn before the low-norm mask and PCA truncation existed, and it has never
been re-tested with them.

There is a concrete reason to expect the ratio to benefit MORE than d_end does:

    d_start = 1 - cos(z_pert[NH,p], z_orig[NH,p])

is measured after a SINGLE prediction step, so it is tiny -- and tiny cosine distances are
exactly where a short ||z|| blows the value up (corr(||z||, d_end) = -0.641 on static
patches; the effect is worse the smaller the true displacement). Dividing by a noisy
denominator injects that noise multiplicatively. If most of the ratio's disadvantage is
denominator noise, masking low-||z|| patches and truncating to the top PCs should rescue it.
If the ratio is structurally wrong -- reordering patches by how quiet they started -- no
amount of cleanup helps.

Grid, all on identical rollouts:
    subspace   m in {4, 8, 384}      cosine inside the top-m principal components
    mask       k in {0, 30}          drop the k lowest-||z|| patches
    estimator  d_end  vs  ftle = (1/T) log(d_end / d_start)
    reduction  mean / p90 / max over kept patches
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
FLOOR = 1e-3          # noise floor on d_end, as in the production server


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
    ap.add_argument("--out", default="outputs/ftle_with_fixes.json")
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
            zv = z["visual"]
            rows.append({
                "ep": ep, "y": y,
                "zo_s": zv[0, NH].cpu().numpy().astype(np.float32),
                "zo_e": zv[0, -1].cpu().numpy().astype(np.float32),
                "zp_s": zv[1:, NH].cpu().numpy().astype(np.float32),
                "zp_e": zv[1:, -1].cpu().numpy().astype(np.float32),
                "norm": zv[0, NH - 1].norm(dim=-1).cpu().numpy(),
            })
            if i % 50 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")

    bank = np.concatenate([r["zo_e"][keep] for r, y in zip(rows, ys) if y == 0], 0)
    mu = bank.mean(0)
    _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
    print(f"PCA on {bank.shape[0]} safe patch vectors; "
          f"top1={(S**2/(S**2).sum())[0]:.3f} top4={(S**2/(S**2).sum())[:4].sum():.3f}")

    def cos_dist(zp, zo, m):
        """(J,P) cosine distance, optionally inside the top-m PC subspace."""
        if m >= 384:
            num = (zp * zo[None]).sum(-1)
            den = np.linalg.norm(zp, axis=-1) * np.linalg.norm(zo, axis=-1)[None] + 1e-9
        else:
            P = Vt[:m]
            o = (zo - mu) @ P.T; p = (zp - mu) @ P.T
            num = (p * o[None]).sum(-1)
            den = np.linalg.norm(p, axis=-1) * np.linalg.norm(o, axis=-1)[None] + 1e-9
        return 1 - num / den

    cache = {}
    for m in (4, 8, 384):
        cache[m] = [(cos_dist(r["zp_s"], r["zo_s"], m) + 1e-4,
                     cos_dist(r["zp_e"], r["zo_e"], m) + 1e-4) for r in rows]

    def score(ds, de, mask, est, red):
        d_s, d_e = ds[:, mask], de[:, mask]
        if est == "d_end":
            v = d_e.mean(0)
        else:
            lam = (1.0 / NP) * np.log(d_e / d_s)
            lam = np.where(d_e > FLOOR, lam, np.nan)     # production noise floor
            with np.errstate(invalid="ignore"):
                v = np.nanmean(lam, axis=0)
        v = v[np.isfinite(v)]
        if v.size < 4:
            return np.nan
        return {"mean": v.mean(), "p90": np.percentile(v, 90), "max": v.max()}[red]

    print("\n=== does the FTLE ratio survive the fixes? (AUC) ===")
    hdr = f"{'est':<7}{'m':>5}{'k':>4}" + "".join(f"{r:>9}" for r in ("mean", "p90", "max"))
    print(hdr); print("-" * len(hdr))
    best = {"d_end": ("", -1), "ftle": ("", -1)}
    out = {}
    for est in ("d_end", "ftle"):
        for m in (384, 8, 4):
            for k in (0, 30):
                cells = []
                for red in ("mean", "p90", "max"):
                    v = []
                    for r, (ds, de) in zip(rows, cache[m]):
                        msk = keep.copy()
                        if k:
                            msk[np.argsort(r["norm"])[:k]] = False
                        v.append(score(ds, de, msk, est, red))
                    v = np.array(v); a = auc(v[ys == 1], v[ys == 0]); cells.append(a)
                    out[f"{est}_m{m}_k{k}_{red}"] = a
                    if a > best[est][1]:
                        best[est] = (f"m={m}, k={k}, {red}", a)
                print(f"{est:<7}{m:>5}{k:>4}" + "".join(f"{c:>9.3f}" for c in cells))
        print()
    print(f"best d_end : {best['d_end'][0]}  AUC {best['d_end'][1]:.3f}")
    print(f"best ftle  : {best['ftle'][0]}  AUC {best['ftle'][1]:.3f}")
    print(f"gap        : {best['ftle'][1] - best['d_end'][1]:+.3f}")
    print("\nreference: original ftle 0.599, original d_end p90 0.799 (full 1772-chunk eval)")

    # is the denominator's damage mostly noise (fixable) or structural (not)?
    print("\n=== where does the ratio lose? ===")
    for m in (384, 4):
        for k in (0, 30):
            ds_all, de_all = [], []
            for r, (ds, de) in zip(rows, cache[m]):
                msk = keep.copy()
                if k:
                    msk[np.argsort(r["norm"])[:k]] = False
                ds_all.append(ds[:, msk].mean()); de_all.append(de[:, msk].mean())
            ds_all = np.array(ds_all); de_all = np.array(de_all)
            print(f"  m={m:<4} k={k:<3} d_start mean {ds_all.mean():.5f} "
                  f"(CV {ds_all.std()/ds_all.mean():.3f})   "
                  f"d_end mean {de_all.mean():.5f} (CV {de_all.std()/de_all.mean():.3f})   "
                  f"AUC of d_start alone {auc(ds_all[ys==1], ds_all[ys==0]):.3f}")
    print("  if d_start alone has AUC well below 0.5, dividing by it actively destroys signal")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "auc": out,
               "best_d_end": best["d_end"], "best_ftle": best["ftle"]},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
