"""
Is the signal strong at PATCH level and lost in the chunk-level aggregation?

Observation from the dend_p90 videos: background and blank-table patches sit around 0.05
while patches on a toppling block reach 0.1+. If that holds, d_end discriminates spatially
and the problem is how 84 patch values get collapsed into one chunk score.

Two tests:

  1. PATCH-LEVEL AUC. Label each patch by whether the ground-truth image content there
     actually changes over the prediction horizon ("moving") or not ("static"), then ask
     whether d_end ranks moving patches above static ones. This measures the spatial signal
     directly, independent of any chunk-level reduction.

  2. BETTER CHUNK REDUCTIONS. If the spatial signal is real, reductions that exploit it --
     counting patches over a threshold, or requiring spatially adjacent patches to fire
     together -- should beat p90/mean, which average signal into background.
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


def patch_motion(img_a, img_b):
    """Per-patch mean absolute pixel change between two frames."""
    d = np.abs(crop(img_b).astype(np.float32) - crop(img_a).astype(np.float32)).mean(2)
    return d.reshape(GRID, 16, GRID, 16).mean(axis=(1, 3)).reshape(-1)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
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

    chunks = []
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
            motion = patch_motion(imgs[NH], imgs[-1])       # GT change over the horizon
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
            zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
            de = ((1 - torch.nn.functional.cosine_similarity(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4)
            chunks.append((ep, y, de.mean(0).cpu().numpy(), de.cpu().numpy(), motion))
            print(".", end="", flush=True)
    env.close(); print()

    # ---------------- TEST 1: patch-level ----------------
    MOVE = 6.0            # mean abs pixel change marking a patch as genuinely moving
    pos_v, neg_v = [], []
    for ep, y, de_mean, de_full, motion in chunks:
        k = keep
        mv = (motion > MOVE) & k
        st = (motion <= 1.0) & k
        pos_v += list(de_mean[mv]); neg_v += list(de_mean[st])
    print(f"\n=== TEST 1: does d_end find WHERE the motion is? ===")
    print(f"  moving patches (GT change > {MOVE}) : n={len(pos_v):5d}  mean d_end {np.mean(pos_v):.4f}")
    print(f"  static patches (GT change <= 1.0)   : n={len(neg_v):5d}  mean d_end {np.mean(neg_v):.4f}")
    print(f"  ratio {np.mean(pos_v)/max(np.mean(neg_v),1e-9):.2f}x     PATCH-LEVEL AUC = {auc(pos_v, neg_v):.3f}")

    # ---------------- TEST 2: chunk reductions ----------------
    print(f"\n=== TEST 2: chunk-level reductions ({sum(1 for c in chunks if c[1]==1)} unsafe, "
          f"{sum(1 for c in chunks if c[1]==0)} safe) ===")
    def reductions(de_mean, de_full):
        k = keep
        v = de_mean[k]
        grid = np.where(keep, de_mean, 0).reshape(GRID, GRID)
        # spatial coherence: best 2x2 block average
        pool = np.array([[grid[i:i+2, j:j+2].mean() for j in range(GRID-1)] for i in range(GRID-1)])
        return {
            "p90 (current best)": float(np.percentile(v, 90)),
            "mean": float(v.mean()),
            "max": float(v.max()),
            "count > 0.05": float((v > 0.05).sum()),
            "count > 0.10": float((v > 0.10).sum()),
            "count > 0.15": float((v > 0.15).sum()),
            "top5 mean": float(np.sort(v)[-5:].mean()),
            "best 2x2 block": float(pool.max()),
            "best 3x3 block": float(max(grid[i:i+3, j:j+3].mean()
                                        for i in range(GRID-2) for j in range(GRID-2))),
        }
    names = list(reductions(chunks[0][2], chunks[0][3]).keys())
    print(f"{'reduction':<22}{'AUC':>8}{'unsafe mean':>14}{'safe mean':>12}")
    print("-" * 58)
    res = []
    for n in names:
        p = [reductions(c[2], c[3])[n] for c in chunks if c[1] == 1]
        q = [reductions(c[2], c[3])[n] for c in chunks if c[1] == 0]
        a = auc(p, q); res.append((n, a))
        print(f"{n:<22}{a:>8.3f}{np.mean(p):>14.4f}{np.mean(q):>12.4f}")
    b = max(res, key=lambda r: r[1])
    print(f"\nbest reduction: {b[0]}  AUC {b[1]:.3f}")


if __name__ == "__main__":
    main()
