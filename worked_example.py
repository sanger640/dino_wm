"""
A worked example of the estimators on real chunks, plus the boundary question.

Part 1 walks one failure chunk and one success chunk number by number, showing exactly
which (perturbation, patch) pair each estimator selects and why they disagree.

Part 2 addresses whether the variance statistic has a principled zero like FTLE does.
FTLE has one: lambda = (1/T)log(d_end/d_start) > 0 means the perturbation grew. Raw
std(d_end) has no such boundary -- it is a magnitude, always >= 0, with no scale to
compare against. But the same trick that gives FTLE its boundary can be applied to the
ensemble spread:

    lambda_var = (1/T) * log( std_j(d_end) / std_j(d_start) )

i.e. did the SPREAD across perturbations grow or shrink over the horizon. That is
dimensionless, scale-free, and zero exactly when the ensemble neither fans out nor
contracts -- the same interpretation FTLE has, but built on a second moment over 49
samples instead of an extremum over 4116.

This computes it and checks whether the zero actually separates safe from unsafe.
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
NH, NP, N = 3, 8, 50
dev = "cuda"
LMDB = "/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single.lmdb"
LABELS = "/home/sanger/wksp/panda_express/labels_noise50.json"


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


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
keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()


def rollout_chunk(txn, meta, ep, start, seed=0):
    keys = meta["episodes"][ep]["keys"]["cam2"]
    acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
    props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
    raw = [txn.get(keys[start + t].encode()) for t in range(span)]
    if any(r is None for r in raw):
        return None
    imgs = [dec(r) for r in raw]
    vis = torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])).float().to(dev) / 255.
    vis = tf(vis)
    obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
           "proprio": ((torch.from_numpy(props[start:start + NH]).float().to(dev) - pm) / psd).unsqueeze(0).repeat(N, 1, 1)}
    g = torch.Generator(device=dev); g.manual_seed(seed)
    a = torch.from_numpy(acts[start:start + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
    a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * 0.05
    a = (a - am) / asd
    with torch.no_grad():
        z, _ = model.rollout(obs, a)
    zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
    ds = (1 - torch.nn.functional.cosine_similarity(zn[:, NH], zo[:, NH], dim=-1))
    de = (1 - torch.nn.functional.cosine_similarity(zn[:, -1], zo[:, -1], dim=-1))
    return ds.cpu().numpy()[:, keep], de.cpu().numpy()[:, keep]


with env.begin() as txn:
    meta = pickle.loads(txn.get(b"__metadata__"))

    # ---------------------------------------------------------------- PART 1
    fs8 = labels["8"]["failure_step"]
    cases = [("ep 8  UNSAFE (predictive chunk)", "8", (fs8 - NH) // NP * NP),
             ("ep 12 SAFE   (mid-episode)", "12", 56)]
    store = {}
    for title, ep, start in cases:
        ds, de = rollout_chunk(txn, meta, ep, start)
        store[ep] = (ds, de)
        J, P = ds.shape
        lam = (1.0 / NP) * np.log((de + 1e-4) / (ds + 1e-4))
        j, p = np.unravel_index(np.argmax(lam), lam.shape)
        print(f"\n{'='*74}\n{title}   chunk start {start}   arrays: {J} perturbations x {P} patches\n{'='*74}")
        print(f"  d_start : min {ds.min():.5f}  median {np.median(ds):.5f}  max {ds.max():.5f}")
        print(f"  d_end   : min {de.min():.5f}  median {np.median(de):.5f}  max {de.max():.5f}")
        print(f"\n  --- what max-max selects ---")
        print(f"  argmax at perturbation {j}, patch {p}")
        print(f"     d_start there = {ds[j,p]:.6f}   (median over all = {np.median(ds):.6f}) "
              f"-> {np.median(ds)/max(ds[j,p],1e-9):.1f}x SMALLER than typical")
        print(f"     d_end   there = {de[j,p]:.6f}   (median over all = {np.median(de):.6f}) "
              f"-> {de[j,p]/max(np.median(de),1e-9):.1f}x larger than typical")
        print(f"     lambda  there = (1/{NP})*log({de[j,p]:.6f}/{ds[j,p]:.6f}) = {lam[j,p]:.4f}   <== the reported score")
        print(f"\n  --- the same chunk under other reductions ---")
        per_pert_max = lam.max(axis=1)
        print(f"     max over perturbations of (max over patches) = {per_pert_max.max():.4f}   [production]")
        print(f"     MEAN over perturbations of (max over patches) = {per_pert_max.mean():.4f}")
        print(f"     mean d_end (all pert, all patches)            = {de.mean():.6f}")
        print(f"     std of d_end ACROSS perturbations, mean over patches = {de.std(axis=0).mean():.6f}")

    # how often does the ratio promote a quiet-start patch?
    print(f"\n{'='*74}\nWhy the ratio misbehaves\n{'='*74}")
    for ep in ["8", "12"]:
        ds, de = store[ep]
        lam = (1.0 / NP) * np.log((de + 1e-4) / (ds + 1e-4))
        j, p = np.unravel_index(np.argmax(lam), lam.shape)
        pct_ds = (ds < ds[j, p]).mean() * 100
        pct_de = (de < de[j, p]).mean() * 100
        print(f"  ep {ep}: the argmax cell sits at the {pct_ds:5.1f}th percentile of d_start "
              f"and the {pct_de:5.1f}th of d_end")
    print("  (a low d_start percentile means the ratio is selecting cells that started quiet,")
    print("   i.e. where the perturbation had barely propagated -- the least informative cells)")

    # ---------------------------------------------------------------- PART 2
    print(f"\n{'='*74}\nPART 2: is there a principled zero for the variance?\n{'='*74}")
    targets, ns = [], 0
    for ep, v in labels.items():
        if ep not in meta["episodes"]:
            continue
        keys = meta["episodes"][ep]["keys"]["cam2"]
        if v["outcome"] == "failure":
            s = v["failure_step"] - NH - NP + 1
            s = max(0, (s // NP) * NP)
            if s + span < len(keys):
                targets.append((ep, s, 1))
        elif ns < 16:
            for s in (len(keys) // 3, 2 * len(keys) // 3):
                s = (s // NP) * NP
                if s + span < len(keys):
                    targets.append((ep, s, 0))
            ns += 1

    rows = []
    for ep, s, y in targets:
        r = rollout_chunk(txn, meta, ep, s)
        if r is None:
            continue
        ds, de = r
        lam_ftle = (1.0 / NP) * np.log((de + 1e-4) / (ds + 1e-4))
        floor = de > 1e-3
        ftle_prod = np.where(floor, lam_ftle, -np.inf).max()
        # ensemble spread ratio: did the spread ACROSS perturbations grow over the horizon?
        s_start = ds.std(axis=0)          # (P,) spread across perturbations at step 1
        s_end = de.std(axis=0)            # (P,) at step T
        lam_var = (1.0 / NP) * np.log((s_end + 1e-6) / (s_start + 1e-6))
        rows.append((ep, y, float(ftle_prod), float(np.mean(lam_var)), float(np.max(lam_var)),
                     float(s_end.mean())))
        print(".", end="", flush=True)
    print()

    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    y = arr[:, 0]
    names = ["ftle_prod (max-max)", "lambda_var (mean patch)", "lambda_var (max patch)", "std d_end (raw)"]
    print(f"\n{len(rows)} chunks: {int(y.sum())} unsafe, {int((1-y).sum())} safe\n")
    print(f"{'statistic':<26}{'unsafe mean':>13}{'safe mean':>11}{'AUC':>7}{'  frac>0 unsafe':>16}{'frac>0 safe':>13}")
    print("-" * 88)
    for i, nm in enumerate(names):
        col = arr[:, i + 1]
        pos, neg = col[y == 1], col[y == 0]
        pos_f = pos[np.isfinite(pos)]; neg_f = neg[np.isfinite(neg)]
        auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos_f for b in neg_f])
        f1 = (pos_f > 0).mean(); f0 = (neg_f > 0).mean()
        print(f"{nm:<26}{pos_f.mean():>13.4f}{neg_f.mean():>11.4f}{auc:>7.3f}{f1:>16.0%}{f0:>13.0%}")
    print("\n  frac>0 = fraction of chunks the statistic declares 'expanding' at its natural zero.")
    print("  A usable boundary needs this HIGH for unsafe and LOW for safe.")
env.close()
