"""
Does masking patches by CONTENT beat masking them by geometry?

The row mask (rows 2-7) is a fixed rectangle. But what makes a patch useless is that it
carries no task signal -- a blank stretch of table -- and that is a property of the image,
not of a fixed row range. Measured earlier:

    correlation(patch content, FTLE)  = +0.004   <- FTLE is content-BLIND
    correlation(patch content, d_end) = +0.205   <- d_end prefers content, weakly

Content-blindness is fatal under a max over 84 patches: if every patch scores about the same
regardless of what is in it, the argmax is decided by noise. This tests masking patches whose
observed content falls below a threshold, computed per chunk from the last observation frame,
so it adapts as the arm and blocks move.

Compares, on identical cached rollouts:
    geometric   rows 2-7 only (current production mask)
    content     occupancy above a percentile of the chunk's own patches
    both        intersection
across the estimator family, ranked by failure/success AUC.
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
N_SUCCESS = 25


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def crop(im, s=224):
    h, w = im.shape[:2]; sc = s / min(h, w)
    im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    sh, sw = im.shape[:2]
    return im[(sh - s) // 2:(sh + s) // 2, (sw - s) // 2:(sw + s) // 2]


def occupancy(img):
    """Per-patch pixel std of the observation -- 0 on blank table, high on texture."""
    g = crop(img).astype(np.float32).mean(2)
    return g.reshape(GRID, 16, GRID, 16).std(axis=(1, 3)).reshape(-1)


def estimators(ds, de, keep):
    if keep.sum() < 4:
        return None
    de_k, ds_k = de[:, keep], ds[:, keep]
    lam = (1.0 / NP) * np.log(de_k / ds_k)
    floor = de_k > 1e-3
    per_pert = np.where(floor, lam, -np.inf).max(axis=1)
    fin = np.isfinite(per_pert)
    return {
        "dend_mean": float(de_k.mean()),
        "dend_p90": float(np.percentile(de_k, 90)),
        "dend_std": float(de_k.std(axis=0).mean()),
        "dend_maxpatch_meanpert": float(de_k.max(axis=1).mean()),
        "ftle": float(per_pert[fin].max()) if fin.any() else np.nan,
        "ftle_maxpatch_meanpert": float(per_pert[fin].mean()) if fin.any() else np.nan,
    }


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
    span = NH + NP
    geo = build_patch_keep_mask(196, torch.device("cpu")).numpy()

    cache = []
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
            elif ns < N_SUCCESS:
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
            occ = occupancy(imgs[NH - 1])            # last observed frame
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
            ds = ((1 - torch.nn.functional.cosine_similarity(zn[:, NH], zo[:, NH], dim=-1)) + 1e-4).cpu().numpy()
            de = ((1 - torch.nn.functional.cosine_similarity(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).cpu().numpy()
            cache.append((ep, y, ds, de, occ))
            print(".", end="", flush=True)
    env.close(); print()

    y_all = np.array([c[1] for c in cache])
    print(f"\n{len(cache)} chunks, {int(y_all.sum())} positive, {int((1-y_all).sum())} negative")

    def make_mask(occ, kind, pct):
        if kind == "geometric":
            return geo.copy()
        thr = np.percentile(occ, pct)
        content = occ > thr
        return content if kind == "content" else (content & geo)

    schemes = [("geometric", None)] + \
              [(k, p) for k in ("content", "both") for p in (40, 60, 75)]
    names = ["dend_mean", "dend_p90", "dend_std", "dend_maxpatch_meanpert",
             "ftle", "ftle_maxpatch_meanpert"]
    print(f"\n{'mask':<20}{'kept':>6}" + "".join(f"{n:>25}" for n in names))
    print("-" * (26 + 25 * len(names)))
    best = (None, -1)
    for kind, pct in schemes:
        cols, kept = {}, []
        for ep, y, ds, de, occ in cache:
            m = make_mask(occ, kind, pct)
            kept.append(m.sum())
            v = estimators(ds, de, m)
            if v is None:
                continue
            for n in names:
                cols.setdefault(n, []).append((y, v[n]))
        row = []
        for n in names:
            arr = cols[n]
            pos = np.array([s for yy, s in arr if yy == 1 and np.isfinite(s)])
            neg = np.array([s for yy, s in arr if yy == 0 and np.isfinite(s)])
            auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]) if len(pos) and len(neg) else np.nan
            row.append(auc)
            if auc > best[1]:
                best = (f"{kind}{'' if pct is None else f' p{pct}'} / {n}", auc)
        lbl = kind if pct is None else f"{kind} p{pct}"
        print(f"{lbl:<20}{int(np.mean(kept)):>6}" + "".join(f"{a:>25.3f}" for a in row))
    print(f"\nbest: {best[0]}  AUC {best[1]:.3f}")
    print("geometric = current production mask (rows 2-7)")
    print("content   = per-chunk occupancy above the given percentile of that chunk's patches")


if __name__ == "__main__":
    main()
