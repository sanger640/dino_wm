"""
Fair comparison of correct-foreground vs correct-background PC1 masking.

Section 7.33 compared both signs at a FIXED 75% keep -- a percentage that was originally
selected via held-out cross-validation for the BACKGROUND-keeping (buggy) direction
(pca_mask_heldout.py picked 75% in 20/20 splits under that sign). It was never re-tuned for
the foreground direction, so that comparison may have been unfair to the foreground side.

This sweeps keep-percentage separately for EACH sign, with proper held-out selection (pick
% on the train half, score on the test half, 20 episode splits) -- reusing the same
checkpointed rollouts as pc1_mask_full_corpus.py and pc1_sign_fix_and_rerun.py, so no GPU
pass is needed.

Also tries a simple ensemble: z-score each direction's score across the safe distribution
and take the max, to see whether foreground and background carry complementary information
that neither captures alone.
"""
import argparse, json, pickle
from pathlib import Path

import cv2, lmdb, numpy as np

LMDB = "/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"
NH, NP, GRID = 3, 8, 14
span = NH + NP
PCTS = (10, 25, 40, 50, 60, 75, 90, 100)
N_SPLITS = 20


def build_keep_mask():
    keep = np.ones(196, dtype=bool)
    for r in (0, 1, 8, 9, 10, 11, 12, 13):
        keep[r * GRID:(r + 1) * GRID] = False
    return keep


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
    ap.add_argument("--checkpoint", default="outputs/pc1_mask_full_corpus.rows.pkl")
    ap.add_argument("--out", default="outputs/pc1_sign_sweep_compare.json")
    args = ap.parse_args()

    print(f"loading checkpoint {args.checkpoint} ...", flush=True)
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    rows, ys, eps_u = ckpt["rows"], ckpt["ys"], ckpt["eps_u"]
    keep = build_keep_mask()
    print(f"{len(rows)} chunks | {int(ys.sum())} unsafe | {len(eps_u)} episodes")

    bank = np.concatenate([r["z_obs"][keep] for r in rows if r["y"] == 0], 0)
    mu = bank.mean(0)
    _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
    print(f"PCA refit: top1 var {(S**2/(S**2).sum())[0]:.3f}")

    # confirmed in section 7.33: -1 = keep true foreground (moving), +1 = keep background
    SIGNS = {"background (sign=+1)": +1, "foreground (sign=-1)": -1}

    def score(v, mask, red="p90"):
        x = v[mask]; x = x[np.isfinite(x)]
        if x.size < 4:
            return np.nan
        return {"mean": x.mean(), "p90": np.percentile(x, 90), "max": x.max()}[red]

    def pc1_mask(z_obs, sign_val, pct):
        pc1 = ((z_obs - mu) @ Vt[0]) * sign_val
        m = keep.copy()
        if pct < 100:
            m &= pc1 >= np.percentile(pc1[keep], 100 - pct)
        return m

    # precompute every (sign, pct) score for every chunk once -- cheap, numpy only
    print("precomputing scores for every (sign, pct, metric) combination...", flush=True)
    cache = {}
    for sign_name, sign_val in SIGNS.items():
        for pct in PCTS:
            for metkey in ("d_end", "ftle_variance"):
                v = np.array([score(r[metkey], pc1_mask(r["z_obs"], sign_val, pct))
                             for r in rows])
                cache[(sign_name, pct, metkey)] = v

    results = {}
    for metkey in ("d_end", "ftle_variance"):
        print(f"\n{'='*70}\nmetric = {metkey}\n{'='*70}")
        for sign_name in SIGNS:
            print(f"\n--- {sign_name}, {N_SPLITS} held-out splits, % selected on train ---")
            print(f"{'split':>7}{'pct picked':>12}{'held-out AUC':>14}")
            recs = []
            for rep in range(N_SPLITS):
                r = np.random.default_rng(300 + rep); sh = list(eps_u); r.shuffle(sh)
                tr_eps, te_eps = set(sh[:len(sh)//2]), set(sh[len(sh)//2:])
                tr_idx = [i for i, x in enumerate(rows) if x["ep"] in tr_eps]
                te_idx = [i for i, x in enumerate(rows) if x["ep"] in te_eps]
                best_pct, best_a = PCTS[0], -1
                for pct in PCTS:
                    v = cache[(sign_name, pct, metkey)][tr_idx]
                    a = auc(v[ys[tr_idx] == 1], v[ys[tr_idx] == 0])
                    if a > best_a:
                        best_a, best_pct = a, pct
                v_te = cache[(sign_name, best_pct, metkey)][te_idx]
                a_te = auc(v_te[ys[te_idx] == 1], v_te[ys[te_idx] == 0])
                recs.append({"pct": best_pct, "train_auc": best_a, "test_auc": a_te})
            ta = np.array([x["test_auc"] for x in recs])
            from collections import Counter
            pcts_chosen = Counter(x["pct"] for x in recs)
            print(f"  held-out AUC: mean {ta.mean():.3f}  median {np.median(ta):.3f}  "
                  f"std {ta.std():.3f}")
            print(f"  pct chosen: {dict(pcts_chosen)}")
            results[f"{metkey}/{sign_name}"] = {
                "mean": float(ta.mean()), "median": float(np.median(ta)),
                "std": float(ta.std()), "pct_chosen": {str(k): v for k, v in pcts_chosen.items()}}

    # ---------------- ensemble: max of z-scored foreground and background ----------------
    print(f"\n{'='*70}\nENSEMBLE: z-scored max(foreground, background)\n{'='*70}")
    for metkey in ("d_end", "ftle_variance"):
        # use each direction's own best FIXED pct (mode of what was chosen above) for the
        # ensemble base scores, then z-score against the full safe distribution and take max
        bg_pct = 75  # established default
        fg_counts = results[f"{metkey}/foreground (sign=-1)"]["pct_chosen"]
        fg_pct = int(max(fg_counts, key=fg_counts.get))
        v_bg = cache[("background (sign=+1)", bg_pct, metkey)]
        v_fg = cache[("foreground (sign=-1)", fg_pct, metkey)]
        safe = ys == 0
        mu_bg, sd_bg = np.nanmean(v_bg[safe]), np.nanstd(v_bg[safe]) + 1e-9
        mu_fg, sd_fg = np.nanmean(v_fg[safe]), np.nanstd(v_fg[safe]) + 1e-9
        z_bg = (v_bg - mu_bg) / sd_bg
        z_fg = (v_fg - mu_fg) / sd_fg
        z_ens = np.nanmax(np.stack([z_bg, z_fg]), axis=0)
        a_bg = auc(v_bg[ys == 1], v_bg[ys == 0])
        a_fg = auc(v_fg[ys == 1], v_fg[ys == 0])
        a_ens = auc(z_ens[ys == 1], z_ens[ys == 0])
        print(f"  {metkey}: background(pct={bg_pct}) AUC={a_bg:.3f}   "
              f"foreground(pct={fg_pct}) AUC={a_fg:.3f}   ensemble-max AUC={a_ens:.3f}")
        results[f"{metkey}/ensemble"] = {"auc": float(a_ens), "bg_pct": bg_pct, "fg_pct": fg_pct}

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for metkey in ("d_end", "ftle_variance"):
        bg = results[f"{metkey}/background (sign=+1)"]["mean"]
        fg = results[f"{metkey}/foreground (sign=-1)"]["mean"]
        ens = results[f"{metkey}/ensemble"]["auc"]
        print(f"  {metkey:<16} background(tuned)={bg:.3f}  foreground(tuned)={fg:.3f}  "
              f"ensemble={ens:.3f}")
    print("\n  reference: low-norm mask (section 7.2) = 0.848")
    print("  reference: background @ fixed 75% (section 7.29/7.33) = 0.894/0.895")
    print("  reference: foreground @ fixed 75% (section 7.33) = 0.714/0.743")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
