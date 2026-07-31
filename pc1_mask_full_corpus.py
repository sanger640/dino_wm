"""
The PC1-mask number that can actually be cited: full 1772-chunk corpus, not the 225-chunk
subsample pca_mask_heldout.py used for comparability with pca_mask_combo.py's original
in-sample result.

Also includes ftle_variance (section 7.23, AUC 0.858, never masked before), since it is
free to add once the rollouts are already being computed -- both scores are reduced from the
SAME N=50 rollout per chunk.

Three masks compared under the SAME held-out procedure as section 7.2 (20 episode splits,
mask parameters selected on train, scored on test):
    unmasked        geometric row-mask only
    low-norm (k=30) the VALIDATED mask from section 7.2, mean 0.848 held-out on d_end p90
    PC1 (signed)    keep-percentage selected per split; foreground side determined from
                    data (higher mean ||z||), matching pca_mask_combo.py's fixed method

Memory note: caching the raw (J=49, P=196, F=384) perturbed-latent tensor for all 1772
chunks would be ~26 GB. Instead, each chunk's contribution is reduced to a (P,) vector
(mean over the 49 perturbations) for BOTH d_end and ftle_variance immediately after the
rollout, and only that plus z_obs (for norm/PC1) is kept. ~1772 x (196x384 + 2x196) floats,
under 1 GB total.
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
    ap.add_argument("--out", default="outputs/pc1_mask_full_corpus.json")
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
    env = lmdb.open(LMDB, readonly=True, lock=False)

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        targets = []
        for ep, v in labels.items():
            if ep not in meta["episodes"]:
                continue
            keys = meta["episodes"][ep]["keys"]["cam2"]
            f = v["failure_step"] if v["outcome"] == "failure" else None
            for s in range(0, len(keys) - span, NP):
                lo, hi = s + NH, s + span - 1
                if f is not None and f < lo:
                    break
                targets.append((ep, s, 1 if (f is not None and f <= hi) else 0))
        print(f"{len(targets)} chunks ({sum(t[2] for t in targets)} unsafe) -- full corpus",
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
            g = torch.Generator(device=dev); g.manual_seed(s)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
            a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                               ).unsqueeze(0).repeat(N, 1, 1)}
            with torch.no_grad():
                z, _ = model.rollout(obs, (a - am) / asd)
            zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
            de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4)
            centroid = zn[:, -1].mean(0, keepdim=True)
            fv = (1 - cs(zn[:, -1], centroid, dim=-1))
            # BUG (fixed): zo = zv[0:1] keeps the full T dimension, so zo[0] is (T,P,F),
            # not (P,F) -- this crashed the analysis after the full 1772-chunk GPU pass had
            # already finished, losing an hour of compute. z_obs must be a SINGLE timestep:
            # the real encoded frame at NH-1, matching every other script's convention
            # (masked_nominal_vs_pert.py, pca_mask_combo.py, pca_mask_heldout.py).
            rows.append({
                "ep": ep, "y": y,
                "d_end": de.mean(0).cpu().numpy().astype(np.float32),          # (P,)
                "ftle_variance": fv.mean(0).cpu().numpy().astype(np.float32),  # (P,)
                "z_obs": zv[0, NH - 1].cpu().numpy().astype(np.float32),       # (P,F)
                "norm": zv[0, NH - 1].norm(dim=-1).cpu().numpy(),              # (P,)
            })
            if i % 100 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    eps_u = sorted({r["ep"] for r in rows})
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe | "
          f"{len(eps_u)} episodes")

    # checkpoint immediately: the GPU pass is the expensive part (~1h); everything below is
    # numpy and should never be allowed to lose it to a bug again
    import pickle as pkl
    ckpt = Path(args.out).with_suffix(".rows.pkl")
    with open(ckpt, "wb") as f:
        pkl.dump({"rows": rows, "ys": ys, "eps_u": eps_u}, f)
    print(f"checkpointed raw rows -> {ckpt}")

    def score(v, mask, red="p90"):
        x = v[mask]; x = x[np.isfinite(x)]
        if x.size < 4:
            return np.nan
        return {"mean": x.mean(), "p90": np.percentile(x, 90), "max": x.max()}[red]

    def auc_masks(idxs_tr, idxs_te, metkey):
        # unmasked
        v_te_un = np.array([score(rows[i][metkey], keep) for i in idxs_te])
        a_un = auc(v_te_un[ys[idxs_te] == 1], v_te_un[ys[idxs_te] == 0])

        # low-norm mask (fixed k=30, the already-validated choice -- no re-selection needed)
        def lm(i):
            m = keep.copy(); m[np.argsort(rows[i]["norm"])[:LOWNORM_K]] = False
            return score(rows[i][metkey], m)
        v_te_lm = np.array([lm(i) for i in idxs_te])
        a_lm = auc(v_te_lm[ys[idxs_te] == 1], v_te_lm[ys[idxs_te] == 0])

        # PC1 mask: fit basis + sign on train, pick % on train, score on test
        bank = np.concatenate([rows[i]["z_obs"][keep] for i in idxs_tr if ys[i] == 0], 0)
        mu = bank.mean(0)
        _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
        pc1_all = np.concatenate([((rows[i]["z_obs"] - mu) @ Vt[0])[keep] for i in idxs_tr])
        nrm_all = np.concatenate([rows[i]["norm"][keep] for i in idxs_tr])
        fg_sign = 1 if nrm_all[pc1_all > np.median(pc1_all)].mean() > \
                       nrm_all[pc1_all <= np.median(pc1_all)].mean() else -1

        def pc1_mask(i, pct):
            pc1 = ((rows[i]["z_obs"] - mu) @ Vt[0]) * fg_sign
            m = keep.copy()
            if pct < 100:
                m &= pc1 >= np.percentile(pc1[keep], 100 - pct)
            return m

        best_pct, best_a = 100, -1
        for pct in PCTS:
            v = np.array([score(rows[i][metkey], pc1_mask(i, pct)) for i in idxs_tr])
            a = auc(v[ys[idxs_tr] == 1], v[ys[idxs_tr] == 0])
            if a > best_a:
                best_a, best_pct = a, pct
        v_te_pc1 = np.array([score(rows[i][metkey], pc1_mask(i, best_pct)) for i in idxs_te])
        a_pc1 = auc(v_te_pc1[ys[idxs_te] == 1], v_te_pc1[ys[idxs_te] == 0])
        return a_un, a_lm, a_pc1, best_pct

    all_results = {}
    for metkey, label in (("d_end", "D_END"), ("ftle_variance", "FTLE_VARIANCE")):
        print(f"\n=== {label}: unmasked vs low-norm(k=30) vs PC1(signed), "
              f"{N_SPLITS} full-corpus episode splits ===")
        print(f"{'split':>7}{'unmasked':>11}{'low-norm':>11}{'PC1':>9}{'PC1 %':>7}")
        print("-" * 47)
        res = []
        for rep in range(N_SPLITS):
            r = np.random.default_rng(300 + rep); sh = list(eps_u); r.shuffle(sh)
            tr_eps, te_eps = set(sh[:len(sh)//2]), set(sh[len(sh)//2:])
            tr_idx = [i for i, x in enumerate(rows) if x["ep"] in tr_eps]
            te_idx = [i for i, x in enumerate(rows) if x["ep"] in te_eps]
            a_un, a_lm, a_pc1, pct = auc_masks(tr_idx, te_idx, metkey)
            res.append({"unmasked": a_un, "low_norm": a_lm, "pc1": a_pc1, "pc1_pct": pct})
            print(f"{rep:>7}{a_un:>11.3f}{a_lm:>11.3f}{a_pc1:>9.3f}{pct:>7}")
        un = np.array([x["unmasked"] for x in res])
        lm = np.array([x["low_norm"] for x in res])
        pc1 = np.array([x["pc1"] for x in res])
        print(f"\n  mean:  unmasked {un.mean():.3f}   low-norm(k=30) {lm.mean():.3f}   "
              f"PC1 {pc1.mean():.3f}")
        print(f"  low-norm - unmasked: {lm.mean()-un.mean():+.3f}   "
              f"PC1 - low-norm: {pc1.mean()-lm.mean():+.3f}")
        if metkey == "d_end":
            print("  reference: section 7.2 low-norm-mask held-out mean = 0.848 (full corpus)")
        all_results[metkey] = res

    # ---------------- operating-point tables, FULL corpus (no split) ----------------
    # matches the convention used everywhere in section 7: thresholds are percentiles of
    # the SAFE-chunk score distribution over the whole corpus, not a held-out half. This is
    # the number to compare against probe/d_end tables elsewhere in the document (they use
    # the same convention). PC1 percentage fixed at 75% -- selected in 20/20 splits both
    # here (pca_mask_heldout.py) and on the small subsample (pca_mask_combo.py).
    PC1_PCT_FIXED = 75
    bank_full = np.concatenate([r["z_obs"][keep] for r in rows if r["y"] == 0], 0)
    mu_f = bank_full.mean(0)
    _, Sf, Vtf = np.linalg.svd(bank_full - mu_f, full_matrices=False)
    pc1_all_f = np.concatenate([((r["z_obs"] - mu_f) @ Vtf[0])[keep] for r in rows])
    nrm_all_f = np.concatenate([r["norm"][keep] for r in rows])
    fg_sign_f = 1 if nrm_all_f[pc1_all_f > np.median(pc1_all_f)].mean() > \
                     nrm_all_f[pc1_all_f <= np.median(pc1_all_f)].mean() else -1

    def mask_for(r, kind):
        if kind == "unmasked":
            return keep
        if kind == "low_norm_k30":
            m = keep.copy(); m[np.argsort(r["norm"])[:LOWNORM_K]] = False
            return m
        if kind == "pc1_75":
            pc1 = ((r["z_obs"] - mu_f) @ Vtf[0]) * fg_sign_f
            m = keep.copy()
            m &= pc1 >= np.percentile(pc1[keep], 100 - PC1_PCT_FIXED)
            return m
        raise ValueError(kind)

    op_tables = {}
    for metkey, label in (("d_end", "D_END"), ("ftle_variance", "FTLE_VARIANCE")):
        for kind in ("unmasked", "low_norm_k30", "pc1_75"):
            v = np.array([score(r[metkey], mask_for(r, kind)) for r in rows])
            safe_v = v[ys == 0]
            print(f"\n=== {label} / {kind}: recall / precision / accuracy / F1 "
                  f"(thresholds = percentile of safe-chunk scores, full corpus) ===")
            print(f"{'thr':>6}{'value':>10}{'TP':>5}{'FP':>6}{'FN':>5}"
                  f"{'recall':>8}{'prec':>7}{'acc':>8}{'F1':>7}")
            rows_out = []
            for q in (75, 80, 85, 90, 95, 99):
                t = np.nanpercentile(safe_v, q)
                pred = v > t
                tp = int((pred & (ys == 1)).sum()); fp = int((pred & (ys == 0)).sum())
                fn = int((~pred & (ys == 1)).sum()); tn = int((~pred & (ys == 0)).sum())
                rec = tp / max(tp + fn, 1); pre = tp / max(tp + fp, 1)
                acc = (tp + tn) / len(ys); f1 = 2 * rec * pre / max(rec + pre, 1e-9)
                rows_out.append({"q": q, "thr": float(t), "recall": rec, "precision": pre,
                                 "accuracy": acc, "f1": f1})
                print(f"{'p'+str(q):>6}{t:>10.4f}{tp:>5}{fp:>6}{fn:>5}"
                      f"{rec:>8.3f}{pre:>7.3f}{acc:>8.4f}{f1:>7.3f}")
            op_tables[f"{metkey}_{kind}"] = rows_out
    base_rate = 1 - ys.mean()
    print(f"\ntrivial 'always safe' accuracy = {base_rate:.4f} (do not read accuracy alone)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_chunks": len(rows), "results": all_results, "operating_points": op_tables},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
