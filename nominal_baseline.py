"""
Does the Deviator Agent earn its cost?

The 49 perturbations exist to answer "is the outcome sensitive to small action errors".
But a single unperturbed rollout already answers a cheaper question: "how much does the
predicted scene change over the horizon". On a 25-episode subset those two scored the
same (0.652 vs 0.676 AUC), which -- if it holds at scale -- means the whole perturbation
apparatus is buying nothing.

    nominal[p]      = 1 - cos( z_orig[NH, p], z_orig[T, p] )       1 rollout,  N=1
    d_end[j, p]     = 1 - cos( z_pert_j[T, p], z_orig[T, p] )      50 rollouts, N=50

Both are extracted from the SAME N=50 rollout so the comparison is exact -- identical
chunks, identical latents, no sampling difference. Timing for the N=1 path is measured
separately.

Every chunk of every episode is scored (stride NP), not a hand-picked chunk per episode,
so the numbers are what the deployed monitor would actually see. Chunks strictly after
the failure step are dropped: the block is already down, and scoring them would reward
detecting the aftermath rather than the event.

Reported at two granularities:
  chunk-level    each (episode, start) is one decision -- what the monitor emits per call
  episode-level  max over an episode's chunks -- "was this run flagged at all"
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


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    n = 0.0
    for a in pos:
        n += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return float(n / (len(pos) * len(neg)))


def reduce_vec(v):
    """Collapse a per-patch vector to scalars. Same family for both methods."""
    v = v[np.isfinite(v)]
    if not len(v):
        return {k: np.nan for k in ("mean", "p90", "p99", "max", "top5", "cnt05")}
    return {"mean": float(v.mean()),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
            "max": float(v.max()),
            "top5": float(np.sort(v)[-5:].mean()),
            "cnt05": float((v > 0.05).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--out", default="outputs/nominal_baseline.json")
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

    rows = []           # (ep, start, y, nominal_vec, d_end_matrix)
    t_n1, t_n50, n_calls = 0.0, 0.0, 0

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = [e for e in meta["episodes"] if e in labels]
        if args.max_episodes:
            eps = eps[:args.max_episodes]
        print(f"{len(eps)} episodes, N={N}, sigma={args.noise_std}")

        for ei, ep in enumerate(eps):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            lab = labels[ep]
            fstep = lab["failure_step"] if lab["outcome"] == "failure" else None
            n = min(len(keys), len(acts), len(props))
            nchunk = 0

            for s in range(0, n - span, NP):
                # prediction horizon covered by this chunk
                lo, hi = s + NH, s + span - 1
                if fstep is not None:
                    if fstep < lo:
                        break                      # already fallen -- stop the episode
                    y = 1 if fstep <= hi else 0
                else:
                    y = 0

                raw = [txn.get(keys[s + i].encode()) for i in range(span)]
                if any(r is None for r in raw):
                    break
                imgs = [dec(r) for r in raw]
                vis = torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])).float().to(dev) / 255.
                vis = tf(vis)
                pro = ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd)
                a0 = torch.from_numpy(acts[s:s + span]).float().to(dev)

                g = torch.Generator(device=dev); g.manual_seed(s)
                a = a0.unsqueeze(0).repeat(N, 1, 1)
                a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
                a = (a - am) / asd
                obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                       "proprio": pro.unsqueeze(0).repeat(N, 1, 1)}

                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    z, _ = model.rollout(obs, a)
                torch.cuda.synchronize(); t_n50 += time.time() - t0

                zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
                d_end = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).cpu().numpy()
                nominal = (1 - cs(zo[0, NH], zo[0, -1], dim=-1)).cpu().numpy()

                # cost of the nominal-only path, measured on its own
                obs1 = {"visual": vis[:NH].unsqueeze(0), "proprio": pro.unsqueeze(0)}
                a1 = ((a0.unsqueeze(0) - am) / asd)
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    model.rollout(obs1, a1)
                torch.cuda.synchronize(); t_n1 += time.time() - t0
                n_calls += 1

                rows.append((ep, s, y, nominal, d_end))
                nchunk += 1
            print(f"[{ei+1}/{len(eps)}] {ep:>4} {lab['outcome']:<7} "
                  f"fstep={str(fstep):<5} chunks={nchunk}", flush=True)
    env.close()

    ys = np.array([r[2] for r in rows])
    print(f"\n{len(rows)} chunks scored | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")
    print(f"latency/chunk: N=50 {1000*t_n50/n_calls:.0f} ms   N=1 {1000*t_n1/n_calls:.0f} ms"
          f"   ({t_n50/max(t_n1,1e-9):.1f}x)")

    methods = {
        "nominal alone   (N=1)": lambda nom, de: nom[keep],
        "d_end mean-pert (N=50)": lambda nom, de: de.mean(0)[keep],
        "d_end std-pert  (N=50)": lambda nom, de: de.std(0)[keep],
    }
    reds = ["mean", "p90", "p99", "max", "top5", "cnt05"]

    cache = {name: [reduce_vec(fn(nom, de)) for _, _, _, nom, de in rows]
             for name, fn in methods.items()}

    print("\n=== CHUNK level (unsafe vs safe), AUC ===")
    print(f"{'method':<24}" + "".join(f"{r:>9}" for r in reds))
    print("-" * (24 + 9 * len(reds)))
    for name in methods:
        cells = [auc([c[r] for c, y in zip(cache[name], ys) if y == 1],
                     [c[r] for c, y in zip(cache[name], ys) if y == 0]) for r in reds]
        print(f"{name:<24}" + "".join(f"{c:>9.3f}" for c in cells))

    # episode level: max over that episode's chunks
    ep_out = sorted({r[0] for r in rows})
    ep_y = {e: 1 if labels[e]["outcome"] == "failure" else 0 for e in ep_out}
    print(f"\n=== EPISODE level (max over chunks), AUC "
          f"[{sum(ep_y.values())} fail / {len(ep_y)-sum(ep_y.values())} ok] ===")
    print(f"{'method':<24}" + "".join(f"{r:>9}" for r in reds))
    print("-" * (24 + 9 * len(reds)))
    for name in methods:
        cells = []
        for r in reds:
            per_ep = {}
            for (e, _, _, _, _), c in zip(rows, cache[name]):
                v = c[r]
                if np.isfinite(v):
                    per_ep[e] = max(per_ep.get(e, -np.inf), v)
            cells.append(auc([v for e, v in per_ep.items() if ep_y[e] == 1],
                             [v for e, v in per_ep.items() if ep_y[e] == 0]))
        print(f"{name:<24}" + "".join(f"{c:>9.3f}" for c in cells))

    # safe-calibrated operating points on the best-of-each
    print("\n=== safe-calibrated thresholds (chunk level, p95 of SAFE chunks) ===")
    print(f"{'method / red':<32}{'thr':>9}{'recall':>9}{'prec':>9}{'F1':>9}")
    print("-" * 68)
    for name in methods:
        for r in ("p90", "max"):
            v = np.array([c[r] for c in cache[name]], float)
            ok = np.isfinite(v)
            thr = np.percentile(v[ok & (ys == 0)], 95)
            pred = ok & (v > thr)
            tp = int((pred & (ys == 1)).sum()); fp = int((pred & (ys == 0)).sum())
            fn = int((~pred & (ys == 1)).sum())
            rec = tp / max(tp + fn, 1); pre = tp / max(tp + fp, 1)
            f1 = 2 * rec * pre / max(rec + pre, 1e-9)
            print(f"{name+' / '+r:<32}{thr:>9.4f}{rec:>9.3f}{pre:>9.3f}{f1:>9.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_chunks": len(rows), "n_unsafe": int(ys.sum()),
               "ms_n50": 1000 * t_n50 / n_calls, "ms_n1": 1000 * t_n1 / n_calls,
               "chunks": [{"ep": e, "start": s, "y": int(y),
                           "nominal": reduce_vec(nom[keep]),
                           "dend_mean": reduce_vec(de.mean(0)[keep]),
                           "dend_std": reduce_vec(de.std(0)[keep])}
                          for e, s, y, nom, de in rows]},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
