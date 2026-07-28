"""
Does the patch mask explain the flat FTLE?

The score is a max over patches. The current mask blanks the top 2 rows (patches 0-27),
but in this scene ALL content sits in grid rows 0-4 -- rows 5-13 are empty table. Empty
patches have near-identical latents, so d_start sits at the 1e-4 epsilon while d_end can
drift just past the 1e-3 floor, making log(d_end/d_start) large from pure background. One
such patch is enough to dominate the max.

This computes per-patch FTLE once per chunk, then re-scores under several masks and reports
the resulting failure/success AUC. If a content-restricted mask lifts AUC well above the
0.588 measured with the current mask, the masking is the problem, not the world model.
"""
import json, pickle
from pathlib import Path

import cv2, hydra, lmdb, numpy as np, torch
from torchvision import transforms

from server_single_max import load_model

AM = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ASD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PM = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PSD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]
NH, NP, N, GRID = 3, 8, 50, 14
SIGMA = 0.05
dev = "cuda"


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def crop224(img, size=224):
    h, w = img.shape[:2]; s = size / min(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = img.shape[:2]
    return img[(sh - size) // 2:(sh + size) // 2, (sw - size) // 2:(sw + size) // 2]


with hydra.initialize(config_path="conf", version_base=None):
    cfg = hydra.compose(config_name="train")
model = load_model(Path("outputs/model_latest_single.pth"), cfg, dev); model.eval()
tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])
am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)

lab = json.load(open("/home/sanger/wksp/panda_express/labels_noise50.json"))
env = lmdb.open("/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single.lmdb",
                readonly=True, lock=False)
span = NH + NP

rows = []          # (kind, per_patch_ftle[196], content_occupancy[196])
with env.begin() as txn:
    meta = pickle.loads(txn.get(b"__metadata__"))
    targets, ns = [], 0
    for ep, v in lab.items():
        if ep not in meta["episodes"]:
            continue
        keys = meta["episodes"][ep]["keys"]["cam2"]
        if v["outcome"] == "failure":
            s = v["failure_step"] - NH - NP + 1
            if 0 <= s and s + span < len(keys):
                targets.append((ep, s, "failure"))
        elif ns < 20:
            s = len(keys) // 2
            if s + span < len(keys):
                targets.append((ep, s, "success")); ns += 1

    for ep, start, kind in targets:
        keys = meta["episodes"][ep]["keys"]["cam2"]
        acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
        props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
        raw = [dec(txn.get(keys[start + t].encode())) for t in range(span)]

        # how much real content does each patch hold? (std over pixels; blank table ~0)
        g = crop224(raw[0]).astype(np.float32).mean(2)
        occ = g.reshape(GRID, 16, GRID, 16).std(axis=(1, 3)).reshape(-1)

        vis = torch.from_numpy(np.stack([np.transpose(r, (2, 0, 1)) for r in raw])).float().to(dev) / 255.
        vis = tf(vis)
        obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
               "proprio": ((torch.from_numpy(props[start:start + NH]).float().to(dev) - pm) / psd).unsqueeze(0).repeat(N, 1, 1)}
        torch.manual_seed(hash(ep) % 10000)
        a = torch.from_numpy(acts[start:start + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
        a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev) * SIGMA
        a = (a - am) / asd
        with torch.no_grad():
            z, _ = model.rollout(obs, a)
        zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
        ds = (1 - torch.nn.functional.cosine_similarity(zn[:, NH], zo[:, NH], dim=-1)) + 1e-4
        de = (1 - torch.nn.functional.cosine_similarity(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4
        lam = (1.0 / NP) * torch.log(de / ds)
        lam[de < 1e-3] = -float("inf")
        rows.append((kind, lam.cpu().numpy(), occ, de.cpu().numpy()))
        print(".", end="", flush=True)
print()

def auc_for(maskfn, name):
    f, s = [], []
    for kind, lam, occ, de in rows:
        m = maskfn(occ)                       # bool[196]: True = keep
        L = lam.copy()
        L[:, ~m] = -np.inf
        v = L.max()
        if not np.isfinite(v):
            v = -1e3
        (f if kind == "failure" else s).append(v)
    f, s = np.array(f), np.array(s)
    a = np.mean([(x > y) + 0.5 * (x == y) for x in f for y in s])
    print(f"{name:<34} fail {f.mean():7.3f}  succ {s.mean():7.3f}  gap {f.mean()-s.mean():+7.3f}  AUC {a:.3f}")

allp = np.arange(GRID * GRID)
print(f"\n{len(rows)} chunks ({sum(1 for r in rows if r[0]=='failure')} failure)")
print(f"{'mask':<34} {'':>7}  {'':>7}  {'':>7}  ")
auc_for(lambda occ: np.ones(196, bool), "no mask")
auc_for(lambda occ: allp >= 28, "current: drop top 2 rows")
auc_for(lambda occ: (allp >= 28) & (allp < 70), "rows 2-4 only (content band)")
auc_for(lambda occ: allp < 70, "rows 0-4 (all content)")
for thr in (2.0, 5.0, 10.0):
    auc_for(lambda occ, t=thr: occ > t, f"occupancy > {thr:g} (non-blank patches)")
auc_for(lambda occ: (occ > 5.0) & (allp >= 28), "occupancy > 5 AND drop top 2 rows")

occ_all = np.stack([r[2] for r in rows]).mean(0)
print("\nmean patch occupancy by grid row (pixel std; ~0 = blank table):")
for r in range(GRID):
    band = occ_all[r * GRID:(r + 1) * GRID]
    print(f"  row {r:>2}: {band.mean():6.2f} {'#' * int(band.mean())}"
          + ("   <- currently masked" if r < 2 else ""))
