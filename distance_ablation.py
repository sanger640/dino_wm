"""
Is cosine the right distance, and are high-norm artifact tokens polluting the score?

Every estimator measured so far uses cosine distance on DINOv2 patch features. That choice
was inherited, never tested. Two independent questions here:

(1) WHICH DISTANCE.
    cosine     1 - cos(a,b)                  current; not a metric (no triangle inequality),
                                             and ~theta^2/2 for small angles, so lambda_cos
                                             = 2 * lambda_angular
    angular    arccos(cos(a,b))              a true metric; monotone in cosine
    chord      sqrt(2*(1-cos))               monotone in cosine -- included as a CONTROL:
                                             patch-level AUC must come out identical to
                                             cosine, chunk-level need not, because mean/p90
                                             over patches is not invariant to monotone maps
    l2         ||a-b||                       keeps norm information cosine discards
    whitened   cosine on (z-mu)/sigma        corrects DINOv2's per-dimension anisotropy

(2) ARTIFACT TOKENS.
    dinov2_vits14 is the register-free variant, which is known to repurpose a few
    low-information background patches into high-norm tokens carrying GLOBAL scene
    information rather than local content (Darcet et al. 2023, "Vision Transformers Need
    Registers"). If those are the blank-table patches that show unexplained divergence,
    then ||z|| identifies them and dropping them is a principled, content-independent mask
    -- unlike the image-occupancy mask, which failed because it keyed on the checkerboard
    texture rather than on task relevance.

    The premise is testable before the fix: correlate ||z|| against d_end among patches the
    ground truth says are STATIC. Positive correlation => artifact hypothesis survives.
    No correlation => hypothesis dead, and this script says so.

Phase 1 estimates whitening statistics from N=1 rollouts on a subset (cheap, ~54 ms/chunk).
Phase 2 does the real N=50 sweep, computing all five distances on identical latents so the
comparison is exact.
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
MOVE_HI, MOVE_LO = 6.0, 1.0          # GT pixel-change thresholds for moving / static


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


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    neg = np.sort(neg)
    r = np.searchsorted(neg, pos, "left") + 0.5 * (
        np.searchsorted(neg, pos, "right") - np.searchsorted(neg, pos, "left"))
    return float(r.mean() / len(neg))


def all_distances(zn, zo, mu, sd):
    """zn (J,P,F) perturbed, zo (1,P,F) original -> dict of (J,P) distances."""
    cs = torch.nn.functional.cosine_similarity
    c = cs(zn, zo, dim=-1).clamp(-1.0, 1.0)
    zw_n, zw_o = (zn - mu) / sd, (zo - mu) / sd
    return {
        "cosine":   1 - c,
        "angular":  torch.arccos(c),
        "chord":    torch.sqrt((2 * (1 - c)).clamp_min(0)),
        "l2":       (zn - zo).norm(dim=-1),
        "whitened": 1 - cs(zw_n, zw_o, dim=-1).clamp(-1.0, 1.0),
    }


def targets_for(meta, labels, eps):
    """(episode, start, label) for every pre-failure chunk."""
    out = []
    for ep in eps:
        keys = meta["episodes"][ep]["keys"]["cam2"]
        lab = labels[ep]
        f = lab["failure_step"] if lab["outcome"] == "failure" else None
        for s in range(0, len(keys) - NH - NP, NP):
            lo, hi = s + NH, s + NH + NP - 1
            if f is not None and f < lo:
                break
            out.append((ep, s, 1 if (f is not None and lo <= f <= hi) else 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--stat-episodes", type=int, default=15)
    ap.add_argument("--out", default="outputs/distance_ablation.json")
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

    def load_chunk(txn, ep, s, keys):
        raw = [txn.get(keys[s + i].encode()) for i in range(span)]
        if any(r is None for r in raw):
            return None
        imgs = [dec(r) for r in raw]
        vis = torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])).float().to(dev) / 255.
        return tf(vis), imgs

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = [e for e in meta["episodes"] if e in labels]
        if args.max_episodes:
            eps = eps[:args.max_episodes]

        # ---------- phase 1: whitening statistics from N=1 rollouts ----------
        print(f"phase 1: whitening stats from {args.stat_episodes} episodes (N=1)")
        acc = []
        for ep in eps[:args.stat_episodes]:
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            for s in range(0, min(len(keys), len(acts), len(props)) - span, NP):
                got = load_chunk(txn, ep, s, keys)
                if got is None:
                    break
                vis, _ = got
                obs = {"visual": vis[:NH].unsqueeze(0),
                       "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd).unsqueeze(0)}
                a = ((torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0) - am) / asd)
                with torch.no_grad():
                    z, _ = model.rollout(obs, a)
                acc.append(z["visual"][0, -1].cpu())
            print(".", end="", flush=True)
        A = torch.cat(acc, 0)
        mu = A.mean(0).to(dev); sd = (A.std(0) + 1e-6).to(dev)
        print(f"\n  stats from {A.shape[0]} patch vectors, dim {A.shape[1]}; "
              f"per-dim sd range {sd.min():.3f}-{sd.max():.3f} (anisotropy {sd.max()/sd.min():.1f}x)")

        # ---------- phase 2: full sweep, N=50 ----------
        tg = targets_for(meta, labels, eps)
        print(f"phase 2: {len(tg)} chunks, N={N}")
        rows = []
        for i, (ep, s, y) in enumerate(tg):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            if s + span > min(len(acts), len(props)):
                continue
            got = load_chunk(txn, ep, s, keys)
            if got is None:
                continue
            vis, imgs = got
            motion = patch_motion(imgs[NH], imgs[-1])
            g = torch.Generator(device=dev); g.manual_seed(s)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
            a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
            a = (a - am) / asd
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                               ).unsqueeze(0).repeat(N, 1, 1)}
            with torch.no_grad():
                z, _ = model.rollout(obs, a)
            zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
            de = all_distances(zn[:, -1], zo[:, -1], mu, sd)
            ds = all_distances(zn[:, NH], zo[:, NH], mu, sd)
            rows.append({
                "ep": ep, "s": s, "y": y, "motion": motion,
                # norm of the last REAL encoded frame -- identifies artifact tokens
                "norm_obs": zo[0, NH - 1].norm(dim=-1).cpu().numpy(),
                "norm_pred": zo[0, -1].norm(dim=-1).cpu().numpy(),
                "de": {k: v.mean(0).cpu().numpy() for k, v in de.items()},
                "de_std": {k: v.std(0).cpu().numpy() for k, v in de.items()},
                "ds": {k: v.mean(0).cpu().numpy() for k, v in ds.items()},
            })
            if i % 50 == 0:
                print(f"  [{i}/{len(tg)}] {ep}", flush=True)
    env.close()

    DIST = ["cosine", "angular", "chord", "l2", "whitened"]
    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")

    # ---------- diagnostic: is ||z|| the artifact signal? ----------
    print("\n=== ARTIFACT-TOKEN PREMISE: does ||z|| explain divergence on STATIC patches? ===")
    nrm, dv, mo = [], [], []
    for r in rows:
        m = keep & (r["motion"] <= MOVE_LO)
        nrm += list(r["norm_obs"][m]); dv += list(r["de"]["cosine"][m]); mo += list(r["motion"][m])
    nrm, dv = np.array(nrm), np.array(dv)
    c = np.corrcoef(nrm, dv)[0, 1]
    hi = nrm > np.percentile(nrm, 90)
    print(f"  static patches: n={len(nrm)}")
    print(f"  corr(||z||, d_end)            = {c:+.3f}")
    print(f"  d_end | top-10% norm          = {dv[hi].mean():.4f}")
    print(f"  d_end | bottom-90% norm       = {dv[~hi].mean():.4f}   "
          f"ratio {dv[hi].mean()/max(dv[~hi].mean(),1e-9):.2f}x")
    print("  -> premise survives only if the correlation is clearly positive")

    # ---------- patch-level: which distance localises motion best ----------
    print("\n=== PATCH level: AUC of d_end against GROUND-TRUTH motion ===")
    print("  (chord is monotone in cosine -- identical here is the CONTROL, not a finding)")
    for d in DIST:
        p, n = [], []
        for r in rows:
            mv = keep & (r["motion"] > MOVE_HI); st = keep & (r["motion"] <= MOVE_LO)
            p += list(r["de"][d][mv]); n += list(r["de"][d][st])
        print(f"  {d:<10} AUC {auc(p, n):.3f}   moving {np.mean(p):.4f}  static {np.mean(n):.4f}")

    # ---------- chunk-level: distance x reduction x artifact mask ----------
    def chunk_score(r, d, red, drop_k):
        m = keep.copy()
        if drop_k:
            order = np.argsort(-r["norm_obs"])
            m[order[:drop_k]] = False
        v = r["de"][d][m]
        v = v[np.isfinite(v)]
        if len(v) < 4:
            return np.nan
        return {"mean": v.mean(), "p90": np.percentile(v, 90), "max": v.max()}[red]

    print("\n=== CHUNK level: AUC (unsafe vs safe) ===")
    print("  drop_k = number of highest-||z|| patches removed before scoring")
    hdr = f"{'distance':<10}{'drop_k':>7}" + "".join(f"{r:>9}" for r in ("mean", "p90", "max"))
    print(hdr); print("-" * len(hdr))
    best = ("", -1)
    for d in DIST:
        for k in (0, 5, 10, 20):
            cells = []
            for red in ("mean", "p90", "max"):
                v = np.array([chunk_score(r, d, red, k) for r in rows])
                a = auc(v[ys == 1], v[ys == 0]); cells.append(a)
                if a > best[1]:
                    best = (f"{d} / {red} / drop{k}", a)
            print(f"{d:<10}{k:>7}" + "".join(f"{c:>9.3f}" for c in cells))
    print(f"\nbest: {best[0]}  AUC {best[1]:.3f}")
    print("baseline to beat: cosine / p90 / drop0 = 0.799 (production mask, 100-ep eval)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows), "corr_norm_dend_static": float(c),
               "rows": [{"ep": r["ep"], "s": r["s"], "y": r["y"],
                         "de": {k: v.tolist() for k, v in r["de"].items()},
                         "norm_obs": r["norm_obs"].tolist(),
                         "motion": r["motion"].tolist()} for r in rows]},
              open(args.out, "w"))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
