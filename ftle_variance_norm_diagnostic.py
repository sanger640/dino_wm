"""
Does ftle_variance suffer the same low-||z|| noise-amplification mechanism as d_end?

Section 7.2's original diagnostic found corr(||z||, d_end) = -0.641 on GROUND-TRUTH STATIC
patches: cosine distance divides out the vector's length, so a near-featureless patch's
direction is poorly determined and wobbles under any perturbation, real or not -- inflating
the score on patches where nothing physically happened. The PC1 mask's improvement on
ftle_variance was explained by analogy to this mechanism, but never checked directly. This
runs the identical methodology (patch_motion from content_mask_study.py, same STATIC
threshold) against ftle_variance instead of d_end.

  corr near 0 or positive => ftle_variance does NOT share the mechanism; PC1's gain on it
                              comes from something else (plausibly still content-relevance,
                              just not via this specific pathway)
  corr clearly negative    => confirms the analogy: PC1 masking helps ftle_variance for the
                              same structural reason it helped d_end
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
MOVE_HI, MOVE_LO = 6.0, 1.0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-safe", type=int, default=200)
    ap.add_argument("--out", default="outputs/ftle_variance_norm_diagnostic.json")
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
            imgs = [dec(r) for r in raw]
            motion = patch_motion(imgs[NH], imgs[-1])
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
            zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
            centroid = zn[:, -1].mean(0, keepdim=True)
            fv = (1 - cs(zn[:, -1], centroid, dim=-1)).mean(0).cpu().numpy()   # (P,)
            de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).mean(0).cpu().numpy()  # (P,)
            nrm = zo[0, NH - 1].norm(dim=-1).cpu().numpy()
            rows.append({"ep": ep, "y": y, "motion": motion, "norm": nrm,
                         "ftle_variance": fv, "d_end": de})
            if i % 50 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    print(f"\n{len(rows)} chunks scored")

    # ---------- correlation on GROUND-TRUTH STATIC patches ----------
    print("\n=== corr(||z||, score) on STATIC patches (ground-truth motion <= 1.0) ===")
    for key in ("ftle_variance", "d_end"):
        nrm_all, sc_all = [], []
        for r in rows:
            static = (r["motion"] <= MOVE_LO) & keep
            nrm_all.append(r["norm"][static]); sc_all.append(r[key][static])
        nrm_all = np.concatenate(nrm_all); sc_all = np.concatenate(sc_all)
        c = np.corrcoef(nrm_all, sc_all)[0, 1]
        hi = nrm_all > np.percentile(nrm_all, 90)
        print(f"  {key:<16} n={len(nrm_all):6d}  corr={c:+.3f}   "
              f"score|top-10%norm={sc_all[hi].mean():.4f}  "
              f"score|bottom-90%norm={sc_all[~hi].mean():.4f}   "
              f"ratio={sc_all[hi].mean()/max(sc_all[~hi].mean(),1e-9):.2f}x")
    print("\n  section 7.2 reference for d_end: corr = -0.641 (Jul 27-28 measurement)")

    # ---------- does ftle_variance ALSO localise ground-truth motion well? ----------
    print("\n=== patch-level AUC of score vs GROUND-TRUTH MOTION (moving>6.0 vs static<=1.0) ===")
    def auc(p, n):
        p = np.asarray(p, float); n = np.sort(np.asarray(n, float))
        if not len(p) or not len(n): return float("nan")
        r = np.searchsorted(n, p, "left") + 0.5 * (np.searchsorted(n, p, "right") - np.searchsorted(n, p, "left"))
        return float(r.mean() / len(n))
    for key in ("ftle_variance", "d_end"):
        mv, st = [], []
        for r in rows:
            m = (r["motion"] > MOVE_HI) & keep; s_ = (r["motion"] <= MOVE_LO) & keep
            mv += list(r[key][m]); st += list(r[key][s_])
        print(f"  {key:<16} AUC {auc(mv, st):.3f}   moving {np.mean(mv):.4f}  static {np.mean(st):.4f}")
    print("  section 7.9 reference for d_end: AUC 0.957-0.960")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows)}, open(args.out, "w"))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
