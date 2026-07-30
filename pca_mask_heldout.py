"""
(3) Held-out validation of the PCA x low-norm mask combination (best cell 0.876, in-sample).

pca_mask_combo.py found m=4/k=30 -> p90 AUC 0.876 on a 225-chunk subset, but the winning
cell was picked from a 16-cell (m,k) grid scored on the SAME data -- optimistic by
construction, the same issue the low-norm mask alone had before section 7.2's split-half
check. This repeats that check: episodes split in half 20 times, the (m,k) grid searched
on the train half only (PCA basis also refit on that half's SAFE chunks), the winning cell
scored on the untouched test half.

(4) Held-out validation of the (corrected, signed) PC1 foreground mask.

pca_mask_combo.py's signed-PC1 result (0.817 at 75% keep) was also read off the same data
it was picked from. Same split-half procedure, selecting the keep-percentage on train and
scoring on test.

Rollouts are cached once (this is the expensive part); everything downstream is numpy, so
20 splits x 16 PCA cells x 5 PC1 percentages costs almost nothing extra.
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
MS = (4, 8, 16, 384)
KS = (0, 10, 20, 30)
PCTS = (100, 75, 50, 40, 25)
N_SPLITS = 20


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
    ap.add_argument("--out", default="outputs/pca_mask_heldout.json")
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
                "z_end": zv[0, -1].cpu().numpy().astype(np.float32),
                "zp_end": zv[1:, -1].cpu().numpy().astype(np.float32),
                "z_obs": zv[0, NH - 1].cpu().numpy().astype(np.float32),
                "norm": zv[0, NH - 1].norm(dim=-1).cpu().numpy(),
            })
            if i % 50 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    eps_u = sorted({r["ep"] for r in rows})
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe | "
          f"{len(eps_u)} unique episodes")

    def fit_pca(idxs):
        bank = np.concatenate([rows[i]["z_end"][keep] for i in idxs if ys[i] == 0], 0)
        mu = bank.mean(0)
        _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
        return mu, Vt

    def d_end_pca(r, mu, Vt, m):
        """Returns one value per PATCH: mean over perturbations first, matching the
        convention used everywhere else in this session (masked_nominal_vs_pert.py,
        robust_ftle.py, metric_family_eval.py) -- NOT the flatten-over-(J,P) reduction
        pca_mask_combo.py used, which pools ~4000 (perturbation,patch) values into one
        percentile instead of ~54-84 patch values and reads very differently."""
        if m >= 384:
            c = (r["zp_end"] * r["z_end"][None]).sum(-1)
            den = np.linalg.norm(r["zp_end"], axis=-1) * np.linalg.norm(r["z_end"]) + 1e-9
            d = 1 - c / den
        else:
            P = Vt[:m]
            o = (r["z_end"] - mu) @ P.T; p = (r["zp_end"] - mu) @ P.T
            num = (p * o[None]).sum(-1)
            den = np.linalg.norm(p, axis=-1) * np.linalg.norm(o) + 1e-9
            d = 1 - num / den
        return d.mean(0)                                        # (J,P) -> (P,)

    def score(d, mask, red="p90"):
        v = d[mask]; v = v[np.isfinite(v)]
        if v.size < 4:
            return np.nan
        return {"mean": v.mean(), "p90": np.percentile(v, 90), "max": v.max()}[red]

    def auc_for(idxs, fn):
        v = np.array([fn(rows[i]) for i in idxs])
        yy = ys[idxs]
        return auc(v[yy == 1], v[yy == 0])

    print(f"\n=== (3) PCA x low-norm mask, {N_SPLITS} episode splits ===")
    print(f"{'':>18}{'train-picked':>14}{'held-out AUC':>14}{'unmasked (384,k0)':>20}")
    print("-" * 66)
    pca_results = []
    for rep in range(N_SPLITS):
        r = np.random.default_rng(100 + rep); sh = list(eps_u); r.shuffle(sh)
        tr_eps, te_eps = set(sh[:len(sh)//2]), set(sh[len(sh)//2:])
        tr_idx = np.array([i for i, x in enumerate(rows) if x["ep"] in tr_eps])
        te_idx = np.array([i for i, x in enumerate(rows) if x["ep"] in te_eps])
        mu, Vt = fit_pca(tr_idx)
        cache_de = {m: {i: d_end_pca(rows[i], mu, Vt, m) for i in np.concatenate([tr_idx, te_idx])}
                    for m in MS}
        best, best_a = (384, 0), -1
        for m in MS:
            for k in KS:
                v = []
                for i in tr_idx:
                    msk = keep.copy()
                    if k:
                        msk[np.argsort(rows[i]["norm"])[:k]] = False
                    v.append(score(cache_de[m][i], msk))
                a = auc(np.array(v)[ys[tr_idx] == 1], np.array(v)[ys[tr_idx] == 0])
                if a > best_a:
                    best_a, best = a, (m, k)
        m, k = best
        v_te = []
        for i in te_idx:
            msk = keep.copy()
            if k:
                msk[np.argsort(rows[i]["norm"])[:k]] = False
            v_te.append(score(cache_de[m][i], msk))
        a_te = auc(np.array(v_te)[ys[te_idx] == 1], np.array(v_te)[ys[te_idx] == 0])
        v_base = [score(cache_de[384][i], keep) for i in te_idx]
        a_base = auc(np.array(v_base)[ys[te_idx] == 1], np.array(v_base)[ys[te_idx] == 0])
        pca_results.append({"m": m, "k": k, "train_auc": best_a, "test_auc": a_te,
                            "baseline_auc": a_base})
        print(f"split {rep:>2}   m={m:<4}k={k:<3}{best_a:>8.3f}{a_te:>14.3f}{a_base:>20.3f}")

    ta = np.array([r["test_auc"] for r in pca_results])
    ba = np.array([r["baseline_auc"] for r in pca_results])
    print(f"\n  held-out AUC: mean {ta.mean():.3f}  median {np.median(ta):.3f}  "
          f"std {ta.std():.3f}")
    print(f"  unmasked (384,k=0) baseline on same splits: mean {ba.mean():.3f}")
    print(f"  gain over baseline: {ta.mean()-ba.mean():+.3f}")
    from collections import Counter
    print(f"  (m,k) chosen: {Counter((r['m'], r['k']) for r in pca_results)}")
    print("  in-sample reference (not held out): m=4,k=30 -> 0.876 (pca_mask_combo.py)")

    print(f"\n=== (4) signed PC1 foreground mask, {N_SPLITS} episode splits ===")
    print(f"{'':>18}{'train-picked %':>16}{'held-out AUC':>14}")
    print("-" * 50)
    pc1_results = []
    for rep in range(N_SPLITS):
        r = np.random.default_rng(200 + rep); sh = list(eps_u); r.shuffle(sh)
        tr_eps, te_eps = set(sh[:len(sh)//2]), set(sh[len(sh)//2:])
        tr_idx = np.array([i for i, x in enumerate(rows) if x["ep"] in tr_eps])
        te_idx = np.array([i for i, x in enumerate(rows) if x["ep"] in te_eps])
        mu, Vt = fit_pca(tr_idx)
        pc1_all = np.concatenate([((rows[i]["z_obs"] - mu) @ Vt[0])[keep] for i in tr_idx])
        nrm_all = np.concatenate([rows[i]["norm"][keep] for i in tr_idx])
        fg_sign = 1 if nrm_all[pc1_all > np.median(pc1_all)].mean() > \
                       nrm_all[pc1_all <= np.median(pc1_all)].mean() else -1
        de_full = {i: d_end_pca(rows[i], mu, Vt, 384) for i in np.concatenate([tr_idx, te_idx])}

        def mask_for(i, pct):
            pc1 = ((rows[i]["z_obs"] - mu) @ Vt[0]) * fg_sign
            msk = keep.copy()
            if pct < 100:
                msk &= pc1 >= np.percentile(pc1[keep], 100 - pct)
            return msk

        best_pct, best_a = 100, -1
        for pct in PCTS:
            v = [score(de_full[i], mask_for(i, pct)) for i in tr_idx]
            a = auc(np.array(v)[ys[tr_idx] == 1], np.array(v)[ys[tr_idx] == 0])
            if a > best_a:
                best_a, best_pct = a, pct
        v_te = [score(de_full[i], mask_for(i, best_pct)) for i in te_idx]
        a_te = auc(np.array(v_te)[ys[te_idx] == 1], np.array(v_te)[ys[te_idx] == 0])
        pc1_results.append({"pct": best_pct, "train_auc": best_a, "test_auc": a_te})
        print(f"split {rep:>2}   {best_pct:>14}%{best_a:>10.3f}{a_te:>14.3f}")

    ta2 = np.array([r["test_auc"] for r in pc1_results])
    print(f"\n  held-out AUC: mean {ta2.mean():.3f}  median {np.median(ta2):.3f}  "
          f"std {ta2.std():.3f}")
    print(f"  (pct) chosen: {Counter(r['pct'] for r in pc1_results)}")
    print("  in-sample reference (not held out): 75% keep -> 0.817 (pca_mask_combo.py)")
    print("  compare to low-norm mask held-out mean 0.848 (masked_nominal_vs_pert.py, sec 7.2)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "pca_mask": pca_results, "pc1_mask": pc1_results},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
