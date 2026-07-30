"""
Render the monitor actually firing: probe score vs threshold, at p80 and p99.

The operating-point table says the probe catches 15/15 topples at p80 (3.1 false halts per
episode) and 6/15 at p99 (0.16 false halts). This shows what those rows look like frame by
frame -- when the alarm goes off relative to when the block actually goes over.

Threshold semantics are the same as everywhere else: a percentile of the SAFE chunk score
distribution, computed on held-out episodes only, so nothing is tuned on what it is scored
against. The probe is fit on the first 50 episodes; every rendered episode comes from the
other 50.

Only N=1 rollouts are needed -- the probe never looks at perturbations -- so this is ~40x
cheaper than the divergence pipeline.

Two videos per threshold:
  p99  episodes the tight threshold catches (high precision, low recall)
  p80  episodes the tight threshold MISSES but the loose one catches, which is exactly the
       recall the extra false alarms are buying
"""
import argparse, json, pickle
from pathlib import Path

import cv2, hydra, lmdb, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from server_single_max import load_model, build_patch_keep_mask
from torchvision import transforms

AM = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ASD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PM = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PSD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]
NH, NP = 3, 8
dev = "cuda"
TOPPLE = 45.0


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def ridge_fit(X, y, lam):
    Xb = np.c_[X, np.ones(len(X))]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def ridge_apply(X, w):
    return np.c_[X, np.ones(len(X))] @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/sanger/wksp/panda_express/tasks/jenga_tilt_100/jenga_tilt.lmdb")
    ap.add_argument("--n-fit-episodes", type=int, default=50)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--episodes", default=None,
                    help="comma-separated episode ids; default picks by catch status")
    ap.add_argument("--thresholds", default="p80,p99")
    ap.add_argument("--outdir", default="/home/sanger/wksp/panda_express/results/threshold_videos")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_model(Path("outputs/model_latest_single.pth"), cfg, dev); model.eval()
    tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
    pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)
    keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
    span = NH + NP
    env = lmdb.open(args.lmdb, readonly=True, lock=False)

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = sorted(meta["episodes"], key=lambda e: int(e))
        fit_eps, held = eps[:args.n_fit_episodes], eps[args.n_fit_episodes:]

        def load(ep):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            return keys, acts, props, tilt, min(len(keys), len(acts), len(props), len(tilt))

        def roll(ep, s, keys, acts, props):
            raw = [txn.get(keys[s + j].encode()) for j in range(span)]
            if any(r is None for r in raw):
                return None, None
            imgs = [dec(r) for r in raw]
            vis = tf(torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])
                                      ).float().to(dev) / 255.)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0)
            obs = {"visual": vis[:NH].unsqueeze(0),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                               ).unsqueeze(0)}
            with torch.no_grad():
                z, _ = model.rollout(obs, (a - am) / asd)
            return z["visual"], imgs

        print(f"fitting probe on {len(fit_eps)} episodes (N=1)", flush=True)
        X, Y = [], []
        for ep in fit_eps:
            keys, acts, props, tilt, n = load(ep)
            for s in range(0, n - span, NP):
                zv, _ = roll(ep, s, keys, acts, props)
                if zv is None:
                    break
                X.append(zv[0, -1][keep].mean(0).cpu().numpy())
                Y.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
        w = ridge_fit(np.stack(X), np.array(Y), args.lam)
        print(f"  fit on {len(X)} chunks")

        # ---- score every held-out episode, then calibrate thresholds on SAFE chunks ----
        print(f"scoring {len(held)} held-out episodes", flush=True)
        epdata = {}
        safe_scores = []
        for ep in held:
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            starts, scores, gts, frames = [], [], [], {}
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                zv, imgs = roll(ep, s, keys, acts, props)
                if zv is None:
                    break
                sc = float(ridge_apply(zv[0, -1][keep].mean(0).cpu().numpy()[None], w)[0])
                starts.append(s); scores.append(sc); frames[s] = imgs
                gts.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
                if not (fstep is not None and lo <= fstep <= hi):
                    safe_scores.append(sc)
            epdata[ep] = dict(starts=starts, scores=scores, gts=gts, frames=frames,
                              fstep=fstep, tilt=tilt)

        # thresholds are built from whatever --thresholds asks for, plus the p80/p99 pair
        # the default episode-selection logic needs
        tags = [t.strip() for t in args.thresholds.split(",")]
        THR = {t: float(np.percentile(safe_scores, float(t.lstrip("p"))))
               for t in set(tags) | {"p80", "p99"}}
        print(f"thresholds from {len(safe_scores)} safe chunks: "
              + ", ".join(f"{t}={THR[t]:.2f} deg" for t in sorted(THR)))

        def fires_at(ep, thr):
            d = epdata[ep]
            for s, sc in zip(d["starts"], d["scores"]):
                if sc > thr:
                    return s
            return None

        fails = [e for e in held if epdata[e]["fstep"] is not None and epdata[e]["starts"]]
        caught99 = [e for e in fails if fires_at(e, THR["p99"]) is not None]
        only80 = [e for e in fails
                  if fires_at(e, THR["p99"]) is None and fires_at(e, THR["p80"]) is not None]
        print(f"{len(fails)} held-out failures | p99 catches {len(caught99)} | "
              f"p80-only catches {len(only80)}")

        if args.episodes:
            want = [e.strip() for e in args.episodes.split(",")]
            sel = [e for e in want if e in epdata and epdata[e]["starts"]]
            missing = [e for e in want if e not in sel]
            if missing:
                print(f"  skipping (not held out / no chunks): {missing}")
            jobs = [(e, t) for e in sel for t in tags]
        else:
            jobs = ([(e, "p99") for e in caught99[:2]]
                    + [(e, "p80") for e in (only80 or caught99)[:2]])
        print(f"rendering {len(jobs)} videos")

        for ep, tag in jobs:
            d = epdata[ep]; thr = THR[tag]
            fire = fires_at(ep, thr)
            starts, scores, gts = d["starts"], d["scores"], d["gts"]
            # a success episode has no fstep, so an alarm there is a false positive
            # with no lead time to report
            lead = (d["fstep"] - fire) if (fire is not None and d["fstep"] is not None) else None
            # name by what actually happened: firing on a success is a FALSE ALARM,
            # not a "catch"
            if d["fstep"] is None:
                outcome = "FALSEALARM" if fire is not None else "ok-silent"
            else:
                outcome = "CAUGHT" if fire is not None else "MISSED"
            vid = out / f"ep{ep}_{tag}_{outcome}.mp4"
            vw, H = None, 480
            for gi, s in enumerate(starts):
                fired_yet = fire is not None and s >= fire
                for im in d["frames"][s][NH:]:
                    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
                    ax.plot(starts[:gi + 1], scores[:gi + 1], "s-", color="#c62828", lw=2.5,
                            ms=4, label="probe score (predicted tilt, deg)")
                    ax.plot(starts[:gi + 1], gts[:gi + 1], "o--", color="#111", lw=1.8, ms=3,
                            alpha=.7, label="ground-truth tilt")
                    ax.axhline(thr, color="#c62828", ls="--", lw=2)
                    ax.text(0, thr + 1.5, f"{tag} threshold = {thr:.1f}°",
                            color="#c62828", fontsize=9, weight="bold")
                    if fire is not None and s >= fire:
                        ax.axvline(fire, color="#c62828", lw=2, alpha=.6)
                        ax.text(fire, 92, " HALT", color="#c62828", fontsize=9, weight="bold")
                    if d["fstep"] is not None and s >= d["fstep"] - NP:
                        ax.axvline(d["fstep"], color="#111", lw=2, ls=":", alpha=.6)
                        ax.text(d["fstep"], 84, " topple", color="#111", fontsize=9)
                    ax.set_xlim(0, max(starts)); ax.set_ylim(-5, 100)
                    ax.set_xlabel("rollout step"); ax.set_ylabel("tilt (deg)")
                    ax.legend(loc="upper left", fontsize=8)
                    ttl = f"ep {ep} · {tag}"
                    if lead is not None:
                        ttl += f" · HALT {lead} steps before topple"
                    ax.set_title(ttl, fontsize=10)
                    fig.tight_layout(); fig.canvas.draw()
                    plot = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                    plt.close(fig)

                    left = cv2.resize(im, (int(H * im.shape[1] / im.shape[0]), H))
                    band = 46
                    left = cv2.copyMakeBorder(left, band, 0, 0, 0, cv2.BORDER_CONSTANT,
                                              value=(200, 40, 40) if fired_yet else (40, 150, 60))
                    cv2.putText(left, "HALT - UNSAFE" if fired_yet else "running - safe",
                                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    plot = cv2.copyMakeBorder(
                        cv2.resize(plot, (int(H * plot.shape[1] / plot.shape[0]), H)),
                        band, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    canvas = cv2.cvtColor(np.hstack([left, plot]), cv2.COLOR_RGB2BGR)
                    if vw is None:
                        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (canvas.shape[1], canvas.shape[0]))
                    vw.write(canvas)
            if vw is not None:
                vw.release()
            print(f"  {vid.name}: halt at step {fire}, topple at {d['fstep']}, "
                  f"lead {lead} steps", flush=True)
    env.close()
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
