"""
Redo section 7.29's headline result with the sign bug fixed, WITHOUT redoing the expensive
GPU pass.

Bug found by inspecting a rendered video frame: the fg_sign heuristic in every PC1 script
since section 7.15 picked "foreground" as whichever side of PC1 has higher mean ||z||, an
ASSUMPTION that arm/blocks = high norm that was never independently checked. Direct check
against ground-truth motion (jenga_tilt_100, 23520 patches, 30 episodes):
corr(PC1_raw, motion) = -0.479 -- the norm heuristic had it backwards. Visual confirmation:
an annotated frame with the old sign showed the blocks/gripper coloured RED (dropped) and
the blank table GREEN (kept); with the sign flipped, the blocks/gripper are green.

pc1_mask_full_corpus.py already checkpointed its raw per-chunk rows (z_obs, norm, d_end,
ftle_variance for all 1772 chunks) to disk before its own analysis stage ran -- exactly the
lesson from that run's earlier crash. That means the expensive ~1h GPU pass does not need
to be repeated: reload the checkpoint, determine the correct sign for THIS fit (jenga_noise_50
uses a different PCA fit than jenga_tilt_100, so the arbitrary SVD sign convention may differ
between them and must be re-derived here, not assumed from the other dataset) by re-reading
the raw frames for ground-truth motion -- a cheap operation, no model forward pass needed --
then redo the numpy-only held-out AUC splits and operating-point tables with the corrected
sign, alongside the original (buggy) sign for direct comparison.
"""
import argparse, json, pickle
from pathlib import Path

import cv2, lmdb, numpy as np

LMDB = "/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"
NH, NP, GRID = 3, 8, 14
span = NH + NP
LOWNORM_K = 30
N_SPLITS = 20


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def crop(im, s=224):
    h, w = im.shape[:2]; sc = s / min(h, w)
    im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    sh, sw = im.shape[:2]
    return im[(sh - s) // 2:(sh + s) // 2, (sw - s) // 2:(sw + s) // 2]


def patch_motion(a, b):
    d = np.abs(crop(b).astype(np.float32) - crop(a).astype(np.float32)).mean(2)
    return d.reshape(GRID, 16, GRID, 16).mean(axis=(1, 3)).reshape(-1)


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
    ap.add_argument("--out", default="outputs/pc1_sign_fix_results.json")
    args = ap.parse_args()

    print(f"loading checkpoint {args.checkpoint} ...", flush=True)
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    rows, ys, eps_u = ckpt["rows"], ckpt["ys"], ckpt["eps_u"]
    keep = build_keep_mask()
    print(f"{len(rows)} chunks | {int(ys.sum())} unsafe | {len(eps_u)} episodes")

    # ---------- refit mu/Vt exactly as before (deterministic given the same bank) ----------
    bank = np.concatenate([r["z_obs"][keep] for r in rows if r["y"] == 0], 0)
    mu = bank.mean(0)
    _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
    print(f"PCA refit: top1 var {(S**2/(S**2).sum())[0]:.3f}")

    # ---------- reconstruct the exact (ep,s,y) scan order to re-fetch matching frames ----------
    # rows do not store `s`, but the original script's target-building loop is deterministic
    # (same LABELS file, same iteration order), so replaying it recovers the exact alignment.
    # z_obs is already cached -- only the raw frame PAIR needs re-fetching, which is cheap
    # (image decode only, no model forward pass).
    labels = json.load(open("/home/sanger/wksp/panda_express/labels_noise100.json"))
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
                targets.append((ep, s))
        assert len(targets) >= len(rows), (
            f"scan-order reconstruction mismatch: {len(targets)} targets vs {len(rows)} rows "
            "-- do not trust the sign result below without fixing this first")
        targets = targets[:len(rows)]  # original run may have dropped a few missing chunks

        print("re-reading raw frames for ground-truth motion (no model needed)...", flush=True)
        rng = np.random.default_rng(0)
        sample = rng.choice(len(rows), min(600, len(rows)), replace=False)
        pc1_raw_all, motion_all = [], []
        for i in sample:
            ep, s = targets[i]
            keys = meta["episodes"][ep]["keys"]["cam2"]
            raw = [txn.get(keys[s + j].encode()) for j in range(span)]
            if any(r is None for r in raw):
                continue
            imgs = [dec(r) for r in raw]
            motion = patch_motion(imgs[NH], imgs[-1])
            pc1_raw = (rows[i]["z_obs"] - mu) @ Vt[0]
            pc1_raw_all.append(pc1_raw[keep]); motion_all.append(motion[keep])
    env.close()
    pc1_raw_all = np.concatenate(pc1_raw_all); motion_all = np.concatenate(motion_all)
    corr_motion = float(np.corrcoef(pc1_raw_all, motion_all)[0, 1])
    # positive corr => high raw PC1 already means high motion (foreground) => keep sign +1
    # negative corr => high raw PC1 means LOW motion (background) => flip to -1
    # (matches the earlier standalone check on jenga_tilt_100: corr=-0.479 -> sign=-1,
    # verified visually against the rendered frame)
    correct_sign = 1 if corr_motion > 0 else -1
    print(f"corr(PC1_raw, ground-truth motion) on jenga_noise_50 = {corr_motion:+.3f} "
          f"(n={len(pc1_raw_all)} patches from {len(sample)} chunks)")
    print(f"  -> correct fg_sign for THIS fit = {correct_sign}")

    def score(v, mask, red="p90"):
        x = v[mask]; x = x[np.isfinite(x)]
        if x.size < 4:
            return np.nan
        return {"mean": x.mean(), "p90": np.percentile(x, 90), "max": x.max()}[red]

    def run_config(sign_val, label):
        def pc1_mask(z_obs):
            pc1 = ((z_obs - mu) @ Vt[0]) * sign_val
            m = keep.copy()
            m &= pc1 >= np.percentile(pc1[keep], 25)
            return m

        results = {}
        for metkey in ("d_end", "ftle_variance"):
            print(f"\n=== {label} / {metkey}: {N_SPLITS} held-out episode splits ===")
            print(f"{'split':>7}{'held-out AUC':>14}")
            aucs = []
            for rep in range(N_SPLITS):
                r = np.random.default_rng(300 + rep); sh = list(eps_u); r.shuffle(sh)
                te_eps = set(sh[len(sh) // 2:])
                te_idx = [i for i, x in enumerate(rows) if x["ep"] in te_eps]
                v = np.array([score(rows[i][metkey], pc1_mask(rows[i]["z_obs"])) for i in te_idx])
                yy = ys[te_idx]
                a = auc(v[yy == 1], v[yy == 0])
                aucs.append(a)
            aucs = np.array(aucs)
            print(f"  mean {aucs.mean():.3f}  median {np.median(aucs):.3f}  std {aucs.std():.3f}")
            results[metkey] = {"mean": float(aucs.mean()), "median": float(np.median(aucs)),
                               "std": float(aucs.std()), "splits": aucs.tolist()}
        return results

    print("\n" + "=" * 70)
    buggy = run_config(+1, "BUGGY sign (norm heuristic, +1 -- section 7.29's original)")
    print("\n" + "=" * 70)
    fixed = run_config(correct_sign, f"CORRECTED sign ({correct_sign:+d}, motion-derived)")

    print("\n" + "=" * 70)
    print("=== SUMMARY: buggy vs corrected sign, full 1772-chunk corpus ===")
    for metkey in ("d_end", "ftle_variance"):
        b, f = buggy[metkey]["mean"], fixed[metkey]["mean"]
        print(f"  {metkey:<16} buggy(+1)={b:.3f}   corrected({correct_sign:+d})={f:.3f}   "
              f"diff={f-b:+.3f}")
    print("\n  reference: low-norm mask (section 7.2) = 0.848 held-out")
    print("  reference: buggy-sign PC1 (section 7.29) = 0.894 (d_end) / 0.896 (ftle_variance)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"corr_motion": corr_motion, "correct_sign": correct_sign,
               "buggy_sign_results": buggy, "corrected_sign_results": fixed},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
