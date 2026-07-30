"""
Motion-normalised instability: does dividing out expected motion separate unsafe from safe?

d_end localises motion almost perfectly (patch-level AUC 0.957 against ground-truth image
change) but barely separates unsafe from safe at chunk level (~0.70), because safe chunks
contain motion too -- the arm sweeps, the gripper closes, blocks get nudged without falling.

So d_end answers "is something moving here", not "is the outcome uncertain here". This adds
a denominator with a real physical scale:

    nominal_change[p] = 1 - cos( z_orig[NH,p], z_orig[T,p] )   how much patch p evolves
                                                                under the true action alone
    spread[p]         = std_j( d_end[j,p] )                     do the 49 perturbations
                                                                disagree about the outcome
    normalized[p]     = spread[p] / (nominal_change[p] + eps)

A patch the arm sweeps past scores high on both, so it normalises down. A patch where a block
sits near its balance point changes little nominally but the perturbations scatter it, so it
normalises up.

Falsifiable prediction: normalized should score LOWER than d_end on patch-level AUC vs motion
(motion is divided out by construction) and HIGHER than 0.695 on chunk-level AUC vs the safety
label. If chunk AUC does not move, the hypothesis is wrong.
"""
import json, pickle
from pathlib import Path

import cv2, hydra, lmdb, numpy as np, torch
from torchvision import transforms

from server_single_max import load_model, build_patch_keep_mask

AM = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ASD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PM = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PSD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]
NH, NP, N, GRID = 3, 8, 50, 14
EPS = 1e-3
dev = "cuda"
LMDB = "/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"
LABELS = "/home/sanger/wksp/panda_express/labels_noise100.json"


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
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def main():
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
    rows = []

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        targets, ns = [], 0
        for ep, v in labels.items():
            if ep not in meta["episodes"]:
                continue
            keys = meta["episodes"][ep]["keys"]["cam2"]
            if v["outcome"] == "failure":
                s = max(0, ((v["failure_step"] - NH - NP + 1) // NP) * NP)
                if s + span < len(keys):
                    targets.append((ep, s, 1))
            elif ns < 25:
                for s in (len(keys) // 3, 2 * len(keys) // 3):
                    s = (s // NP) * NP
                    if s + span < len(keys):
                        targets.append((ep, s, 0))
                ns += 1

        for ep, s, y in targets:
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            raw = [txn.get(keys[s + i].encode()) for i in range(span)]
            if any(r is None for r in raw):
                continue
            imgs = [dec(r) for r in raw]
            motion = patch_motion(imgs[NH], imgs[-1])
            vis = torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])).float().to(dev) / 255.
            vis = tf(vis)
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd).unsqueeze(0).repeat(N, 1, 1)}
            g = torch.Generator(device=dev); g.manual_seed(s)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
            a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * 0.05
            a = (a - am) / asd
            with torch.no_grad():
                z, _ = model.rollout(obs, a)
            zv = z["visual"]
            zo, zn = zv[0:1], zv[1:]
            cs = torch.nn.functional.cosine_similarity
            d_end = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).cpu().numpy()      # (J,P)
            # how much the nominal rollout's own latent moves over the horizon
            nominal = (1 - cs(zo[0, NH], zo[0, -1], dim=-1)).cpu().numpy()              # (P,)
            rows.append((ep, y, d_end, nominal, motion))
            print(".", end="", flush=True)
    env.close(); print()

    print(f"\n{len(rows)} chunks, {sum(r[1] for r in rows)} unsafe, {sum(1-r[1] for r in rows)} safe")

    # ---------- patch-level: does each quantity track MOTION? ----------
    MOVE = 6.0
    p_de, n_de, p_no, n_no = [], [], [], []
    for ep, y, d_end, nominal, motion in rows:
        de_m = d_end.mean(0)
        norm = d_end.std(0) / (nominal + EPS)
        mv = (motion > MOVE) & keep
        st = (motion <= 1.0) & keep
        p_de += list(de_m[mv]); n_de += list(de_m[st])
        p_no += list(norm[mv]); n_no += list(norm[st])
    print("\n=== patch level: correlation with GROUND-TRUTH MOTION ===")
    print(f"  d_end       moving {np.mean(p_de):.4f}  static {np.mean(n_de):.4f}   AUC {auc(p_de, n_de):.3f}")
    print(f"  normalized  moving {np.mean(p_no):.4f}  static {np.mean(n_no):.4f}   AUC {auc(p_no, n_no):.3f}")
    print("  (normalized SHOULD drop here -- motion is divided out by construction)")

    # ---------- chunk-level: does each separate UNSAFE from SAFE? ----------
    def reduce_all(vec):
        v = vec[keep]
        v = v[np.isfinite(v)]
        if not len(v):
            return {}
        return {"mean": float(v.mean()), "p90": float(np.percentile(v, 90)),
                "p99": float(np.percentile(v, 99)), "max": float(v.max()),
                "count>q90": float((v > np.percentile(v, 90)).sum()),
                "top5": float(np.sort(v)[-5:].mean())}

    variants = {
        "d_end (current)":        lambda de, no: de.mean(0),
        "d_end std":              lambda de, no: de.std(0),
        "normalized (std/nom)":   lambda de, no: de.std(0) / (no + EPS),
        "normalized (mean/nom)":  lambda de, no: de.mean(0) / (no + EPS),
        "nominal alone":          lambda de, no: np.broadcast_to(no, de.shape[1:]).copy(),
    }
    print("\n=== chunk level: UNSAFE vs SAFE ===")
    reds = ["mean", "p90", "p99", "max", "count>q90", "top5"]
    print(f"{'variant':<24}" + "".join(f"{r:>12}" for r in reds))
    print("-" * (24 + 12 * len(reds)))
    best = ("", -1)
    for name, fn in variants.items():
        cells = []
        for r in reds:
            pos = [reduce_all(fn(d, n)).get(r, np.nan) for e, y, d, n, m in rows if y == 1]
            neg = [reduce_all(fn(d, n)).get(r, np.nan) for e, y, d, n, m in rows if y == 0]
            a = auc(pos, neg); cells.append(a)
            if a > best[1]:
                best = (f"{name} / {r}", a)
        print(f"{name:<24}" + "".join(f"{c:>12.3f}" for c in cells))
    print(f"\nbest: {best[0]}  AUC {best[1]:.3f}")
    print("baseline to beat: count>0.05 = 0.695, p90 = 0.676")


if __name__ == "__main__":
    main()
