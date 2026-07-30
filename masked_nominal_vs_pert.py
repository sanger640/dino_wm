"""
Does the Deviator Agent survive once cosine's low-norm noise floor is removed?

Two established results are in tension:

  * nominal alone (N=1, no perturbations) matched the full N=50 apparatus, AUC ~0.82 vs
    ~0.83, at 1/38th the cost -- all bootstrap CIs straddling zero
  * masking the lowest-||z|| patches lifted the perturbation score 0.759 -> 0.851
    (CI [+0.042, +0.149]), because cosine divides by ||z|| and so amplifies noise exactly
    where the DINOv2 feature is weakest

The mask might rescue the perturbations by cleaning up what was drowning them, or it might
lift nominal equally -- in which case the Deviator Agent's case gets worse, not better.
Cached data cannot answer this: nominal_baseline.json stored reduced scalars only.

Both quantities come from the SAME N=50 rollout, so the comparison is exact:

    nominal[p]  = 1 - cos( z_orig[NH,p], z_orig[T,p] )      row 0 only
    d_end[j,p]  = 1 - cos( z_pert_j[T,p], z_orig[T,p] )     rows 1..49

k is additionally chosen on one half of the episodes and scored on the other, because the
earlier k=20/30 was picked by looking at the same data it was evaluated on.
"""
import argparse, json, pickle, time
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
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--out", default="outputs/masked_nominal_vs_pert.json")
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
    cs = torch.nn.functional.cosine_similarity
    rows = []
    t0 = time.time()

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = [e for e in meta["episodes"] if e in labels]
        if args.max_episodes:
            eps = eps[:args.max_episodes]
        for ei, ep in enumerate(eps):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            lab = labels[ep]
            f = lab["failure_step"] if lab["outcome"] == "failure" else None
            n = min(len(keys), len(acts), len(props))
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if f is not None and f < lo:
                    break
                y = 1 if (f is not None and f <= hi) else 0
                raw = [txn.get(keys[s + i].encode()) for i in range(span)]
                if any(r is None for r in raw):
                    break
                vis = torch.from_numpy(np.stack([np.transpose(dec(r), (2, 0, 1)) for r in raw])
                                       ).float().to(dev) / 255.
                vis = tf(vis)
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
                de = (1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4
                rows.append({
                    "ep": ep, "y": y,
                    "nominal": (1 - cs(zo[0, NH], zo[0, -1], dim=-1)).cpu().numpy(),
                    "de_mean": de.mean(0).cpu().numpy(),
                    "de_std": de.std(0).cpu().numpy(),
                    "norm": zo[0, NH - 1].norm(dim=-1).cpu().numpy(),
                })
            print(f"[{ei+1}/{len(eps)}] {ep} {lab['outcome']}  n={len(rows)}", flush=True)
    env.close()

    ys = np.array([r["y"] for r in rows])
    print(f"\n{len(rows)} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe "
          f"| {time.time()-t0:.0f}s")

    METH = ["nominal", "de_mean", "de_std"]

    def score(r, meth, red, k):
        m = keep.copy()
        if k:
            m[np.argsort(r["norm"])[:k]] = False
        v = r[meth][m]
        v = v[np.isfinite(v)]
        if len(v) < 4:
            return np.nan
        return {"mean": v.mean(), "p90": np.percentile(v, 90), "max": v.max()}[red]

    print("\n=== AUC by method x low-norm mask (k = lowest-||z|| patches dropped) ===")
    hdr = f"{'method':<10}{'k':>4}" + "".join(f"{r:>9}" for r in ("mean", "p90", "max"))
    print(hdr); print("-" * len(hdr))
    for meth in METH:
        for k in (0, 10, 20, 30):
            cells = []
            for red in ("mean", "p90", "max"):
                v = np.array([score(r, meth, red, k) for r in rows])
                cells.append(auc(v[ys == 1], v[ys == 0]))
            print(f"{meth:<10}{k:>4}" + "".join(f"{c:>9.3f}" for c in cells))

    # ---------- paired bootstrap: does the perturbation ensemble beat nominal, masked? ----------
    eps_u = sorted({r["ep"] for r in rows})
    idx = {e: [i for i, r in enumerate(rows) if r["ep"] == e] for e in eps_u}
    rng = np.random.default_rng(0)
    cache = {(m, k): np.array([score(r, m, "p90", k) for r in rows])
             for m in METH for k in (0, 20)}
    print("\n=== paired cluster bootstrap over episodes (p90), 2000 reps ===")
    print(f"{'comparison':<44}{'A':>7}{'B':>7}{'A-B':>8}{'95% CI':>18}{'P(A>B)':>9}")
    print("-" * 93)
    for A_, B_ in [(("de_mean", 20), ("nominal", 20)),
                   (("de_std", 20), ("nominal", 20)),
                   (("nominal", 20), ("nominal", 0)),
                   (("de_mean", 20), ("de_mean", 0))]:
        va, vb = cache[A_], cache[B_]
        a0 = auc(va[ys == 1], va[ys == 0]); b0 = auc(vb[ys == 1], vb[ys == 0])
        ds = []
        for _ in range(2000):
            sel = np.concatenate([idx[e] for e in rng.choice(eps_u, len(eps_u), replace=True)])
            yy = ys[sel]
            if yy.sum() == 0 or (1 - yy).sum() == 0:
                continue
            ds.append(auc(va[sel][yy == 1], va[sel][yy == 0])
                      - auc(vb[sel][yy == 1], vb[sel][yy == 0]))
        ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
        lbl = f"{A_[0]}/k={A_[1]}  vs  {B_[0]}/k={B_[1]}"
        print(f"{lbl:<44}{a0:>7.3f}{b0:>7.3f}{a0-b0:>8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{(ds>0).mean():>9.3f}")

    # ---------- held-out k: the earlier k was chosen on the data it was scored on ----------
    print("\n=== held-out k selection (episodes split in half, 20 random splits) ===")
    for meth in METH:
        gaps = []
        for rep in range(20):
            rr = np.random.default_rng(100 + rep)
            sh = list(eps_u); rr.shuffle(sh)
            tr, te = set(sh[:len(sh) // 2]), set(sh[len(sh) // 2:])
            itr = np.array([i for i, r in enumerate(rows) if r["ep"] in tr])
            ite = np.array([i for i, r in enumerate(rows) if r["ep"] in te])
            best_k, best_a = 0, -1
            for k in (0, 10, 20, 30, 40):
                v = np.array([score(r, meth, "p90", k) for r in rows])
                a = auc(v[itr][ys[itr] == 1], v[itr][ys[itr] == 0])
                if np.isfinite(a) and a > best_a:
                    best_a, best_k = a, k
            v0 = np.array([score(r, meth, "p90", 0) for r in rows])
            vk = np.array([score(r, meth, "p90", best_k) for r in rows])
            a0 = auc(v0[ite][ys[ite] == 1], v0[ite][ys[ite] == 0])
            ak = auc(vk[ite][ys[ite] == 1], vk[ite][ys[ite] == 0])
            if np.isfinite(a0) and np.isfinite(ak):
                gaps.append((best_k, ak, a0))
        ks = [g[0] for g in gaps]
        print(f"  {meth:<9} k chosen (median) {int(np.median(ks)):>3} | "
              f"held-out AUC masked {np.mean([g[1] for g in gaps]):.3f} vs "
              f"unmasked {np.mean([g[2] for g in gaps]):.3f} | "
              f"gain {np.mean([g[1]-g[2] for g in gaps]):+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(rows),
               "rows": [{"ep": r["ep"], "y": r["y"],
                         "nominal": r["nominal"].tolist(),
                         "de_mean": r["de_mean"].tolist(),
                         "de_std": r["de_std"].tolist(),
                         "norm": r["norm"].tolist()} for r in rows]},
              open(args.out, "w"))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
