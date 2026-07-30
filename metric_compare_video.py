"""
Render d_end and FTLE on the SAME episodes, so the two can be compared frame by frame.

Both metrics come from one N=50 rollout per chunk, so nothing differs between the two
videos of an episode except the readout:

    d_end = mean_j (1 - cos(z_pert_j[T,p], z_orig[T,p]))          p90 over kept patches
    ftle  = mean_j (1/T) log(d_end[j,p] / d_start[j,p])           p90 over kept patches
    ftle_pooled = mean_j (1/T) log(d_end[j,p] / median_p d_start)  p90 over kept patches

All use the validated low-norm mask (drop the 30 lowest-||z|| patches). Measured AUCs on this
setup: d_end 0.852, ftle_pooled 0.782, ftle 0.710 -- the gap is what these videos make
visible. ftle_pooled replaces the per-patch denominator with ONE median d_start per chunk,
which was the single change that recovered most of the ratio's lost ground (robust_ftle.py):
per-patch denominator noise, not the ratio form itself, was the dominant cost.

Episodes are the same ten used for the tilt-probe videos, so all three metric families can
be laid side by side on identical rollouts.

Thresholds are p90 of each metric's own SAFE chunk distribution, computed across the
held-out episodes -- each metric is judged at its own scale, never a shared cut.
Ground-truth tilt is drawn for reference; a topple is tilt > 45 deg.
"""
import argparse, pickle
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
LOWNORM_K = 30
FLOOR = 1e-3
EPISODES = ["50", "54", "59", "61", "65", "78", "79", "82", "51", "52"]


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/sanger/wksp/panda_express/tasks/jenga_tilt_100/jenga_tilt.lmdb")
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--outdir", default="/home/sanger/wksp/panda_express/results/metric_videos")
    args = ap.parse_args()
    N = args.n_perturb
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
    cs = torch.nn.functional.cosine_similarity
    env = lmdb.open(args.lmdb, readonly=True, lock=False)

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        held = [e for e in sorted(meta["episodes"], key=lambda x: int(x)) if int(e) >= 50]

        def load(ep):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            return keys, acts, props, tilt, min(len(keys), len(acts), len(props), len(tilt))

        def score_chunk(ep, s, keys, acts, props, want_frames):
            raw = [txn.get(keys[s + j].encode()) for j in range(span)]
            if any(r is None for r in raw):
                return None
            imgs = [dec(r) for r in raw]
            vis = tf(torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])
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
            ds = ((1 - cs(zn[:, NH], zo[:, NH], dim=-1)) + 1e-4).cpu().numpy()
            de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).cpu().numpy()
            nrm = zo[0, NH - 1].norm(dim=-1).cpu().numpy()
            m = keep.copy(); m[np.argsort(nrm)[:LOWNORM_K]] = False
            lam = (1.0 / NP) * np.log(de / ds)
            lam = np.where(de > FLOOR, lam, np.nan)
            # pooled denominator: one median d_start for the whole chunk, over kept patches
            pooled = float(np.median(ds[:, m]))
            lam_pool = (1.0 / NP) * np.log(de / pooled)
            lam_pool = np.where(de > FLOOR, lam_pool, np.nan)
            with np.errstate(invalid="ignore"):
                lam_p = np.nanmean(lam, axis=0)
                lam_pool_p = np.nanmean(lam_pool, axis=0)
            v_de = float(np.percentile(de[:, m].mean(0), 90))
            v_ft = float(np.nanpercentile(lam_p[m], 90))
            v_fp = float(np.nanpercentile(lam_pool_p[m], 90))
            return v_de, v_ft, v_fp, (imgs if want_frames else None)

        # ---- pass 1: safe-chunk distributions across ALL held-out episodes ----
        print(f"calibrating thresholds on {len(held)} held-out episodes", flush=True)
        safe_de, safe_ft, safe_fp = [], [], []
        for i, ep in enumerate(held):
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                if fstep is not None and lo <= fstep <= hi:
                    continue                        # unsafe chunk: excluded from calibration
                r = score_chunk(ep, s, keys, acts, props, False)
                if r is None:
                    break
                safe_de.append(r[0]); safe_ft.append(r[1]); safe_fp.append(r[2])
            if i % 10 == 0:
                print(f"  [{i}/{len(held)}] safe chunks={len(safe_de)}", flush=True)
        THR = {"d_end": float(np.percentile(safe_de, 90)),
               "ftle": float(np.nanpercentile(safe_ft, 90)),
               "ftle_pooled": float(np.nanpercentile(safe_fp, 90))}
        print(f"thresholds (p90 of {len(safe_de)} safe chunks): "
              f"d_end={THR['d_end']:.4f}  ftle={THR['ftle']:.4f}  "
              f"ftle_pooled={THR['ftle_pooled']:.4f}")

        # ---- pass 2: render both metrics for the chosen episodes ----
        for ep in EPISODES:
            if ep not in meta["episodes"]:
                continue
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            starts, v_de, v_ft, v_fp, gts, frames = [], [], [], [], [], {}
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                r = score_chunk(ep, s, keys, acts, props, True)
                if r is None:
                    break
                starts.append(s); v_de.append(r[0]); v_ft.append(r[1])
                v_fp.append(r[2]); frames[s] = r[3]
                gts.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
            if not starts:
                continue

            for tag, vals, colour in (("ftle_pooled", v_fp, "#00838f"),):
                thr = THR[tag]
                fire = next((s for s, v in zip(starts, vals) if np.isfinite(v) and v > thr), None)
                lead = (fstep - fire) if (fire is not None and fstep is not None) else None
                failed = fstep is not None
                status = (("FALSEALARM" if fire is not None else "ok-silent") if not failed
                          else ("CAUGHT" if fire is not None else "MISSED"))
                vid = out / f"ep{ep}_{tag}_{status}.mp4"
                lo_y = min(min(v for v in vals if np.isfinite(v)), thr)
                hi_y = max(max(v for v in vals if np.isfinite(v)), thr)
                pad = 0.15 * (hi_y - lo_y + 1e-6)
                vw, H = None, 480
                for gi, s in enumerate(starts):
                    fired = fire is not None and s >= fire
                    for im in frames[s][NH:]:
                        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
                        ax.plot(starts[:gi + 1], vals[:gi + 1], "s-", color=colour, lw=2.5,
                                ms=4, label=f"{tag} (p90, k=30 mask)")
                        ax.axhline(thr, color=colour, ls="--", lw=2)
                        ax.text(0, thr + pad * .12, f"p90 safe threshold = {thr:.4f}",
                                color=colour, fontsize=8, weight="bold")
                        if fire is not None and s >= fire:
                            ax.axvline(fire, color="#c62828", lw=2, alpha=.6)
                            ax.text(fire, hi_y, " HALT", color="#c62828", fontsize=9,
                                    weight="bold")
                        if fstep is not None and s >= fstep - NP:
                            ax.axvline(fstep, color="#111", lw=2, ls=":", alpha=.6)
                            ax.text(fstep, hi_y - pad * .6, " topple", color="#111", fontsize=9)
                        ax.set_xlim(0, max(starts)); ax.set_ylim(lo_y - pad, hi_y + pad)
                        ax.set_xlabel("rollout step"); ax.set_ylabel(tag)
                        ax2 = ax.twinx()
                        ax2.plot(starts[:gi + 1], gts[:gi + 1], "o--", color="#111", lw=1.4,
                                 ms=3, alpha=.55)
                        ax2.set_ylim(-5, 100); ax2.set_ylabel("ground-truth tilt (deg)",
                                                              color="#555", fontsize=9)
                        ax2.axhline(TOPPLE, color="#888", ls=":", lw=1)
                        ttl = f"ep {ep} · {tag}" + (f" · HALT {lead} steps early"
                                                    if lead is not None else
                                                    (" · NO ALARM" if failed else " · safe run"))
                        ax.set_title(ttl, fontsize=10)
                        ax.legend(loc="upper left", fontsize=8)
                        fig.tight_layout(); fig.canvas.draw()
                        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                        plt.close(fig)

                        left = cv2.resize(im, (int(H * im.shape[1] / im.shape[0]), H))
                        band = 46
                        left = cv2.copyMakeBorder(left, band, 0, 0, 0, cv2.BORDER_CONSTANT,
                                                  value=(200, 40, 40) if fired else (40, 150, 60))
                        cv2.putText(left, f"{tag}: " + ("HALT" if fired else "running"),
                                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
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
                print(f"  {vid.name}: halt={fire} topple={fstep} lead={lead}", flush=True)
    env.close()
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
