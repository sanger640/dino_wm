"""
Three panels per episode, on 10 held-out episodes not used in any earlier video batch:

  1. camera feed, HALT status band for each metric
  2. the SAME frame with a colour-coded 14x14 patch grid:
         black  = dropped by the fixed geometric row-mask (ceiling/floor rows 0,1,8-13)
         green  = kept (foreground by PC1, counts toward the score)
         red    = dropped by PC1 within the row-masked 84 patches (background)
         bright outline = the single WORST patch this chunk (argmax ftle_variance),
                          the same localisation server_single_max.py exposes as
                          max_patch_idx for the original FTLE -- ftle_variance is patch-wise
                          until the p90 reduction, so this costs nothing extra
  3. probe (left axis, degrees) and ftle_variance_pc1 (right axis) plotted together against
     ground-truth tilt, each with its own p90 safe-calibrated threshold and halt marker

  THIS IS THE VALIDATED CONFIGURATION (section 7.29/7.30): PC1 refines WHICH of the
  row-mask's 84 patches matter most, operating INSIDE the geometric prior, not replacing
  it. Section 7.31 showed PC1 alone (no row mask) fails badly on these same 10 episodes
  (2/7 topples caught vs the properly-masked 7/8) -- this run is the direct counterpart on
  the identical episode set for side-by-side comparison.

IMPORTANT what PC1 masking actually does: perturbations act on the ROBOT'S ACTIONS, not on
patches -- every patch is perturbed identically because the whole scene rolls forward under
one perturbed action sequence. PC1 does not choose what gets perturbed. It chooses which
patches' resulting divergence values get COUNTED when the final chunk score is computed --
a foreground filter applied at the READOUT stage, after the 49 perturbed rollouts already
exist. Background patches still get perturbed and still diverge; their result is simply
discarded before the p90 reduction.

Two-pass, mirroring pc1_metric_video.py: pass 1 refits the PC1 basis + probe + calibrates
both thresholds on ALL held-out safe chunks (unavoidable -- the earlier fit was never
persisted); this version saves the fitted basis + thresholds to disk so a future script
does not have to redo it. Pass 2 renders the 10 chosen episodes.
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
NH, NP, GRID = 3, 8, 14
dev = "cuda"
TOPPLE = 45.0
PC1_PCT = 75
EPISODES = ["89", "93", "94", "95", "96", "98", "99", "53", "55", "56"]   # same set as the PC1-alone run


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def crop(im, s=224):
    h, w = im.shape[:2]; sc = s / min(h, w)
    im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    sh, sw = im.shape[:2]
    return im[(sh - s) // 2:(sh + s) // 2, (sw - s) // 2:(sw + s) // 2]


def ridge_fit(X, y, lam):
    Xb = np.c_[X, np.ones(len(X))]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def ridge_apply(X, w):
    return np.c_[X, np.ones(len(X))] @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/sanger/wksp/panda_express/tasks/jenga_tilt_100/jenga_tilt.lmdb")
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-fit-episodes", type=int, default=50)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--outdir", default="/home/sanger/wksp/panda_express/results/pc1_probe_videos_combined")
    ap.add_argument("--basis-out", default="outputs/pc1_probe_basis_combined.pkl")
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
        eps = sorted(meta["episodes"], key=lambda e: int(e))
        fit_eps = eps[:args.n_fit_episodes]
        held = [e for e in eps if int(e) >= args.n_fit_episodes]

        def load(ep):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            return keys, acts, props, tilt, min(len(keys), len(acts), len(props), len(tilt))

        def roll(ep, s, keys, acts, props, n_pert, want_frames):
            raw = [txn.get(keys[s + j].encode()) for j in range(span)]
            if any(r is None for r in raw):
                return None
            imgs = [dec(r) for r in raw]
            vis = tf(torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])
                                      ).float().to(dev) / 255.)
            a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(n_pert, 1, 1)
            if n_pert > 1:
                g = torch.Generator(device=dev); g.manual_seed(s)
                a[1:, :, :3] += torch.randn(n_pert - 1, span, 3, device=dev,
                                            generator=g) * args.noise_std
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(n_pert, 1, 1, 1, 1),
                   "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                               ).unsqueeze(0).repeat(n_pert, 1, 1)}
            with torch.no_grad():
                z, _ = model.rollout(obs, (a - am) / asd)
            return z["visual"], (imgs if want_frames else None)

        # ---- fit probe (N=1, fitting episodes) ----
        print(f"fitting probe on {len(fit_eps)} episodes (N=1)", flush=True)
        X, Y = [], []
        for ep in fit_eps:
            keys, acts, props, tilt, n = load(ep)
            for s in range(0, n - span, NP):
                r = roll(ep, s, keys, acts, props, 1, False)
                if r is None:
                    break
                zv, _ = r
                X.append(zv[0, -1][keep].mean(0).cpu().numpy())
                Y.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
        w = ridge_fit(np.stack(X), np.array(Y), args.lam)
        print(f"  fit on {len(X)} chunks")

        # ---- pass 1: PC1 basis + both thresholds, ALL held-out safe chunks (N=50) ----
        print(f"pass 1: {len(held)} held-out episodes (N={N})", flush=True)
        safe_probe, safe_fv, safe_zobs, safe_norm = [], [], [], []
        for i, ep in enumerate(held):
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                if fstep is not None and lo <= fstep <= hi:
                    continue
                r = roll(ep, s, keys, acts, props, N, False)
                if r is None:
                    break
                zv, _ = r
                zo, zn = zv[0:1], zv[1:]
                centroid = zn[:, -1].mean(0, keepdim=True)
                fv = (1 - cs(zn[:, -1], centroid, dim=-1)).mean(0).cpu().numpy()
                z_obs = zv[0, NH - 1].cpu().numpy().astype(np.float32)
                norm = zv[0, NH - 1].norm(dim=-1).cpu().numpy()
                probe_pred = float(ridge_apply(zv[0, -1][keep].mean(0).cpu().numpy()[None], w)[0])
                safe_probe.append(probe_pred); safe_fv.append(fv)
                safe_zobs.append(z_obs); safe_norm.append(norm)
            if i % 10 == 0:
                print(f"  [{i}/{len(held)}] safe chunks={len(safe_fv)}", flush=True)

        # geometric row mask FIRST (validated config): PC1 is fit only on the already
        # row-filtered 84 patches and refines within them, exactly like pc1_metric_video.py
        bank = np.concatenate([z[keep] for z in safe_zobs], 0)
        mu = bank.mean(0)
        _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
        pc1_all = np.concatenate([((z - mu) @ Vt[0])[keep] for z in safe_zobs])
        nrm_all = np.concatenate([nn[keep] for nn in safe_norm])
        fg_sign = 1 if nrm_all[pc1_all > np.median(pc1_all)].mean() > \
                       nrm_all[pc1_all <= np.median(pc1_all)].mean() else -1

        def pc1_mask(z_obs):
            pc1 = ((z_obs - mu) @ Vt[0]) * fg_sign
            m = keep.copy()
            m &= pc1 >= np.percentile(pc1[keep], 100 - PC1_PCT)
            return m

        def score(v, mask):
            x = v[mask]; x = x[np.isfinite(x)]
            return float(np.percentile(x, 90)) if x.size >= 4 else np.nan

        safe_fv_pc1 = [score(fv, pc1_mask(z)) for fv, z in zip(safe_fv, safe_zobs)]
        THR = {"probe": float(np.percentile(safe_probe, 90)),
               "ftle_variance_pc1": float(np.nanpercentile(safe_fv_pc1, 90))}
        print(f"thresholds: probe={THR['probe']:.2f} deg  "
              f"ftle_variance_pc1={THR['ftle_variance_pc1']:.4f}")

        Path(args.basis_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.basis_out, "wb") as f:
            pickle.dump({"mu": mu, "Vt": Vt, "fg_sign": fg_sign, "thresholds": THR,
                        "probe_weights": w}, f)
        print(f"basis + thresholds saved -> {args.basis_out}")

        # ---- pass 2: render the 10 chosen episodes ----
        for ep in EPISODES:
            if ep not in meta["episodes"]:
                continue
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            starts, v_probe, v_fv, gts, frames, worst_patch, masks = [], [], [], [], {}, [], []
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                r = roll(ep, s, keys, acts, props, N, True)
                if r is None:
                    break
                zv, imgs = r
                zo, zn = zv[0:1], zv[1:]
                centroid = zn[:, -1].mean(0, keepdim=True)
                fv = (1 - cs(zn[:, -1], centroid, dim=-1)).mean(0).cpu().numpy()
                z_obs = zv[0, NH - 1].cpu().numpy().astype(np.float32)
                m = pc1_mask(z_obs)
                probe_pred = float(ridge_apply(zv[0, -1][keep].mean(0).cpu().numpy()[None], w)[0])
                starts.append(s); v_probe.append(probe_pred); v_fv.append(score(fv, m))
                frames[s] = imgs; masks.append(m)
                fv_masked = np.where(m, fv, -np.inf)
                worst_patch.append(int(np.argmax(fv_masked)))
                gts.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
            if not starts:
                continue

            thr_p, thr_f = THR["probe"], THR["ftle_variance_pc1"]
            fire_p = next((s for s, v in zip(starts, v_probe) if v > thr_p), None)
            fire_f = next((s for s, v in zip(starts, v_fv) if np.isfinite(v) and v > thr_f), None)
            lead_p = (fstep - fire_p) if (fire_p is not None and fstep is not None) else None
            lead_f = (fstep - fire_f) if (fire_f is not None and fstep is not None) else None
            failed = fstep is not None

            def tagname(fire):
                if not failed:
                    return "FALSEALARM" if fire is not None else "ok-silent"
                return "CAUGHT" if fire is not None else "MISSED"
            vid = out / f"ep{ep}_probe-{tagname(fire_p)}_fv-{tagname(fire_f)}.mp4"

            lo_p, hi_p = min(v_probe + gts + [thr_p]), max(v_probe + gts + [thr_p, 90])
            fv_fin = [v for v in v_fv if np.isfinite(v)]
            lo_f, hi_f = min(fv_fin + [thr_f]), max(fv_fin + [thr_f])
            padp = 0.1 * (hi_p - lo_p + 1e-6); padf = 0.15 * (hi_f - lo_f + 1e-6)

            vw, H = None, 440
            for gi, s in enumerate(starts):
                fp = fire_p is not None and s >= fire_p
                ff = fire_f is not None and s >= fire_f
                m = masks[gi]; wp = worst_patch[gi]
                for im in frames[s][NH:]:
                    # panel 3: combined plot
                    fig, ax = plt.subplots(figsize=(5.0, 4.6), dpi=100)
                    ax.plot(starts[:gi + 1], v_probe[:gi + 1], "s-", color="#c62828", lw=2,
                            ms=3.5, label="probe (deg)")
                    ax.plot(starts[:gi + 1], gts[:gi + 1], "o--", color="#111", lw=1.3,
                            ms=2.5, alpha=.6, label="GT tilt")
                    ax.axhline(thr_p, color="#c62828", ls="--", lw=1.3, alpha=.7)
                    ax.set_xlim(0, max(starts)); ax.set_ylim(lo_p - padp, hi_p + padp)
                    ax.set_xlabel("rollout step"); ax.set_ylabel("degrees")
                    ax2 = ax.twinx()
                    ax2.plot(starts[:gi + 1], v_fv[:gi + 1], "^-", color="#2e7d32", lw=2,
                             ms=3.5, label="ftle_variance_pc1")
                    ax2.axhline(thr_f, color="#2e7d32", ls="--", lw=1.3, alpha=.7)
                    ax2.set_ylim(lo_f - padf, hi_f + padf)
                    ax2.set_ylabel("ftle_variance_pc1", color="#2e7d32")
                    if fire_p is not None and s >= fire_p:
                        ax.axvline(fire_p, color="#c62828", lw=1.5, alpha=.5)
                    if fire_f is not None and s >= fire_f:
                        ax2.axvline(fire_f, color="#2e7d32", lw=1.5, alpha=.5, ls=":")
                    if fstep is not None and s >= fstep - NP:
                        ax.axvline(fstep, color="#111", lw=2, ls=":", alpha=.6)
                    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
                    ttl = f"ep {ep}"
                    if lead_p is not None: ttl += f" | probe +{lead_p}"
                    if lead_f is not None: ttl += f" | fv_pc1 +{lead_f}"
                    ax.set_title(ttl, fontsize=9)
                    fig.tight_layout(); fig.canvas.draw()
                    plot = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                    plt.close(fig)

                    # panel 2: geometric row mask + PC1 refinement (the validated config)
                    base = cv2.resize(crop(im, 224), (H, H))
                    overlay = base.copy()
                    cell = H // GRID
                    row_mask_keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
                    for p in range(196):
                        r_, c_ = p // GRID, p % GRID
                        x0, y0 = c_ * cell, r_ * cell
                        if not row_mask_keep[p]:
                            col = (30, 30, 30)
                        elif m[p]:
                            col = (40, 180, 60)
                        else:
                            col = (200, 50, 50)
                        cv2.rectangle(overlay, (x0, y0), (x0 + cell, y0 + cell), col, -1)
                    panel2 = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)
                    wr, wc = wp // GRID, wp % GRID
                    cv2.rectangle(panel2, (wc * cell, wr * cell),
                                 (wc * cell + cell, wr * cell + cell), (255, 230, 0), 3)
                    cv2.putText(panel2, "green=foreground(kept) red=background(dropped)",
                                (6, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    cv2.putText(panel2, "black=geometric mask  yellow=worst patch",
                                (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                    # panel 1: camera feed with halt bands
                    left = cv2.resize(im, (int(H * im.shape[1] / im.shape[0]), H))
                    band = 46
                    col1 = (200, 40, 40) if fp else (40, 150, 60)
                    left = cv2.copyMakeBorder(left, band, 0, 0, 0, cv2.BORDER_CONSTANT, value=col1)
                    cv2.putText(left, f"probe: {'HALT' if fp else 'running'}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    col2 = (200, 40, 40) if ff else (40, 150, 60)
                    cv2.putText(left, f"fv_pc1: {'HALT' if ff else 'running'}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                    panel2b = cv2.copyMakeBorder(panel2, band, 0, 0, 0, cv2.BORDER_CONSTANT,
                                                 value=(60, 60, 60))
                    plot_r = cv2.resize(plot, (int(H * plot.shape[1] / plot.shape[0]), H))
                    plot_r = cv2.copyMakeBorder(plot_r, band, 0, 0, 0, cv2.BORDER_CONSTANT,
                                                value=(255, 255, 255))
                    canvas = cv2.cvtColor(np.hstack([left, panel2b, plot_r]), cv2.COLOR_RGB2BGR)
                    if vw is None:
                        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (canvas.shape[1], canvas.shape[0]))
                    vw.write(canvas)
            if vw is not None:
                vw.release()
            print(f"  {vid.name}: probe halt={fire_p} lead={lead_p} | "
                  f"fv_pc1 halt={fire_f} lead={lead_f} | topple={fstep}", flush=True)
    env.close()
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
