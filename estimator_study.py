"""
Is the signal in the rollouts, and is the FTLE estimator throwing it away?

Everything so far varied things AROUND the estimator -- threshold, epsilon, mask, labels.
This varies the estimator itself, on identical rollouts.

For each chunk we roll out once and cache the raw per-patch, per-perturbation divergences:

    d_start[j, p]   1 - cos(z_pert_j, z_orig) at the first PREDICTED step
    d_end  [j, p]   the same at the final predicted step

Every scoring variant is then computed offline from that same cache, so any difference
between them is purely the statistic and not the model, the data, or the noise draw.

Why the current estimator is a suspect:
  * it is a DOUBLE MAXIMUM -- max over 49 perturbations of (max over patches), i.e. an
    extremum over ~4100 numbers per chunk. Extrema are the least stable statistic available.
  * it is a RATIO of small numbers. d_start is the divergence after a single predicted step,
    sitting on a +1e-4 floor, so log(d_end/d_start) can amplify noise rather than signal.

Stage A (--episodes): per-chunk traces through named episodes, to see whether any variant
tracks the fall you can see in the video.
Stage B (--auc): cache many episodes and rank variants by failure/success AUC.

Usage:
    python estimator_study.py --episodes 9 8 12
    python estimator_study.py --auc
"""

import argparse, json, pickle
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", default="/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single.lmdb")
    p.add_argument("--labels", default="/home/sanger/wksp/panda_express/labels_noise50.json")
    p.add_argument("--ckpt", default="outputs/model_latest_single.pth")
    p.add_argument("--episodes", nargs="+", default=None, help="Stage A: trace these episodes")
    p.add_argument("--auc", action="store_true", help="Stage B: cache many episodes and rank variants")
    p.add_argument("--n-success", type=int, default=12, help="Stage B: success episodes to include")
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--repeats", type=int, default=1, help="repeat each chunk with fresh noise draws")
    p.add_argument("--cache", default="results/estimator_cache.npz")
    p.add_argument("--output", default="results/estimator_study.json")
    return p.parse_args()


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def _lam_var(ds, de):
    """(1/T) log(std_j(d_end) / std_j(d_start)) per patch -- ensemble-spread growth rate."""
    s_start = ds.std(axis=0)
    s_end = de.std(axis=0)
    return (1.0 / NP) * np.log((s_end + 1e-6) / (s_start + 1e-6))


# ----------------------------------------------------------------------------- estimators
def estimators(ds, de, keep):
    """ds, de: (J, P) numpy. keep: (P,) bool. Returns {name: scalar}."""
    ds = ds[:, keep]; de = de[:, keep]
    lam = (1.0 / NP) * np.log((de + 1e-4) / (ds + 1e-4))
    floor = de > 1e-3                      # the production noise floor
    lam_f = np.where(floor, lam, -np.inf)

    def safe(x):
        return float(x) if np.isfinite(x) else float("nan")

    per_pert_max = np.where(floor.any(1), np.nanmax(np.where(floor, lam, np.nan), axis=1), np.nan)
    out = {
        # --- current production statistic ---
        "ftle_max_max":        safe(np.nanmax(lam_f)),
        # --- vary the reduction over PERTURBATIONS (patch reduction stays max) ---
        "ftle_max_meanpert":   safe(np.nanmean(per_pert_max)),
        "ftle_max_medpert":    safe(np.nanmedian(per_pert_max)),
        "ftle_max_p90pert":    safe(np.nanpercentile(per_pert_max, 90)),
        # --- vary the reduction over PATCHES (perturbation reduction stays max) ---
        "ftle_meanpatch_max":  safe(np.nanmax(np.nanmean(lam, axis=1))),
        "ftle_top5_max":       safe(np.nanmax(np.sort(lam, axis=1)[:, -5:].mean(1))),
        # --- both means: maximally stable ---
        "ftle_mean_mean":      safe(np.nanmean(lam)),
        # --- drop the ratio: use the end divergence directly ---
        "dend_max_max":        safe(de.max()),
        "dend_max_meanpert":   safe(de.max(axis=1).mean()),
        "dend_meanpatch_max":  safe(de.mean(axis=1).max()),
        "dend_mean_mean":      safe(de.mean()),
        # --- difference instead of ratio ---
        "ddiff_max_max":       safe((de - ds).max()),
        "ddiff_mean_mean":     safe((de - ds).mean()),
        # --- spread across perturbations (no original-trajectory reference) ---
        "dend_std_over_pert":  safe(de.std(axis=0).mean()),
        "dend_p90_over_pert":  safe(np.percentile(de, 90)),
        # --- principled-zero variants: did the ENSEMBLE SPREAD grow over the horizon?
        # lambda_var = (1/T) log( std_j(d_end) / std_j(d_start) ), the FTLE construction
        # applied to a second moment over perturbations instead of an extremum.
        "lambda_var_meanpatch": safe(np.mean(_lam_var(ds, de))),
        "lambda_var_maxpatch":  safe(np.max(_lam_var(ds, de))),
        "lambda_var_medpatch":  safe(np.median(_lam_var(ds, de))),
        # same idea on the MEAN divergence rather than the spread
        "lambda_mean_ratio":    safe(np.mean((1.0 / NP) * np.log(
                                    (de.mean(axis=0) + 1e-6) / (ds.mean(axis=0) + 1e-6)))),
    }
    return out


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_model(Path(args.ckpt), cfg, dev); model.eval()
    tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
    pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)
    labels = json.load(open(args.labels))
    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    span = NH + NP
    keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
    print(f"patch mask: keeping {keep.sum()}/196 (rows 2-7)")

    def run_chunk(txn, ep, start, seed):
        keys = env_meta["episodes"][ep]["keys"]["cam2"]
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
        a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
        a = (a - am) / asd
        with torch.no_grad():
            z, _ = model.rollout(obs, a)
        zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
        ds = (1 - torch.nn.functional.cosine_similarity(zn[:, NH], zo[:, NH], dim=-1))
        de = (1 - torch.nn.functional.cosine_similarity(zn[:, -1], zo[:, -1], dim=-1))
        return ds.cpu().numpy(), de.cpu().numpy()

    with env.begin() as txn:
        env_meta = pickle.loads(txn.get(b"__metadata__"))

        # ---------------------------------------------------------------- Stage A
        if args.episodes:
            names = list(estimators(np.ones((2, 196)), np.ones((2, 196)), keep).keys())
            report = {}
            for ep in args.episodes:
                if ep not in env_meta["episodes"]:
                    continue
                info = labels.get(ep, {})
                fs = info.get("failure_step")
                keys = env_meta["episodes"][ep]["keys"]["cam2"]
                nmax = len(keys) - span - 1
                rows = []
                for start in range(0, nmax + 1, NP):
                    r = run_chunk(txn, ep, start, seed=start)
                    if r is None:
                        break
                    sc = estimators(*r, keep)
                    rows.append((start, sc))
                report[ep] = dict(outcome=info.get("outcome"), failure_step=fs,
                                  chunks=[{"start": s, **v} for s, v in rows])
                tag = f"{info.get('outcome','?')}" + (f", fs={fs}" if fs is not None else "")
                print(f"\n=== ep {ep} ({tag}) ===")
                key_sel = ["ftle_max_max", "ftle_max_meanpert", "ftle_mean_mean",
                           "dend_max_max", "dend_mean_mean", "dend_std_over_pert"]
                print("  " + "start".rjust(5) + "".join(k.rjust(20) for k in key_sel) + "   window")
                for s, v in rows:
                    mark = ""
                    if fs is not None:
                        if fs - NH - NP + 1 <= s <= fs - NH: mark = "  <== PREDICTIVE"
                        elif s > fs - NH: mark = "  (post)"
                    print("  " + f"{s:>5}" + "".join(f"{v[k]:>20.4f}" for k in key_sel) + mark)
            json.dump(report, open(args.output, "w"), indent=2)
            print(f"\n-> {args.output}")

        # ---------------------------------------------------------------- Stage B
        if args.auc:
            cache, meta = [], []
            targets = []
            ns = 0
            for ep, v in labels.items():
                if ep not in env_meta["episodes"]:
                    continue
                if v["outcome"] == "failure":
                    targets.append(ep)
                elif ns < args.n_success:
                    targets.append(ep); ns += 1
            for ep in targets:
                info = labels[ep]; fs = info.get("failure_step")
                keys = env_meta["episodes"][ep]["keys"]["cam2"]
                nmax = len(keys) - span - 1
                for start in range(0, nmax + 1, NP):
                    if info["outcome"] == "failure" and fs is not None and start > fs - NH:
                        continue                       # drop post-failure chunks
                    for rep in range(args.repeats):
                        r = run_chunk(txn, ep, start, seed=start * 97 + rep)
                        if r is None:
                            break
                        lab = 0
                        if info["outcome"] == "failure" and fs is not None:
                            lab = 1 if (fs - NH - NP + 1) <= start <= (fs - NH) else 0
                        cache.append(r); meta.append((ep, start, lab, rep))
                print(".", end="", flush=True)
            print()
            names = list(estimators(cache[0][0], cache[0][1], keep).keys())
            scores = {n: [] for n in names}
            for ds, de in cache:
                v = estimators(ds, de, keep)
                for n in names:
                    scores[n].append(v[n])
            y = np.array([m[2] for m in meta])
            print(f"\n{len(cache)} chunks, {y.sum()} positive, {len(y)-y.sum()} negative")
            print(f"\n{'estimator':<24}{'AUC':>8}{'pos mean':>12}{'neg mean':>12}{'nan':>6}")
            print("-" * 62)
            res = {}
            for n in names:
                s = np.array(scores[n], dtype=float)
                ok = np.isfinite(s)
                pos, neg = s[(y == 1) & ok], s[(y == 0) & ok]
                if len(pos) == 0 or len(neg) == 0:
                    continue
                auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
                res[n] = dict(auc=float(auc), pos=float(pos.mean()), neg=float(neg.mean()),
                              nan=int((~ok).sum()))
                print(f"{n:<24}{auc:>8.3f}{pos.mean():>12.4f}{neg.mean():>12.4f}{(~ok).sum():>6}")
            best = max(res, key=lambda k: res[k]["auc"])
            print(f"\nbest: {best}  AUC {res[best]['auc']:.3f}   (production ftle_max_max: {res['ftle_max_max']['auc']:.3f})")
            json.dump({"meta": [list(m) for m in meta], "results": res,
                       "scores": {n: [float(x) for x in scores[n]] for n in names}},
                      open(args.output, "w"), indent=2)
            print(f"-> {args.output}")
    env.close()


if __name__ == "__main__":
    main()
