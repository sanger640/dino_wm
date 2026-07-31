"""
Render PC1-masked d_end and PC1-masked ftle_variance on the ten episodes used throughout
this session, so the just-validated full-corpus win (section 7.29) can be seen frame by
frame, not just as an AUC number.

Full-corpus held-out results (pc1_mask_full_corpus.py):
    d_end          unmasked 0.756   low-norm(k=30) 0.854   PC1(75%) 0.894
    ftle_variance  unmasked 0.784   low-norm(k=30) 0.838   PC1(75%) 0.896

Both scores come from the SAME N=50 rollout per chunk -- only the readout differs:
    d_end          = mean_j (1 - cos(z_pert_j[T,p], z_orig[T,p]))
    ftle_variance  = mean_j (1 - cos(z_pert_j[T,p], centroid_j(z_pert[T,p])))
                     (spread of the 49 perturbed latents around their own centroid --
                      never references the original/unperturbed trajectory at all)

PC1 mask: fit the PCA basis (mu, Vt) from z_obs (the real encoded frame at NH-1) across the
SAFE chunks of all held-out episodes, sign determined from data (the side with higher mean
||z|| is foreground, matching pc1_mask_full_corpus.py), keep the top 75% by that projection.
This is a TWO-PASS script: pass 1 accumulates the PC1 fitting bank AND every safe chunk's
reduced (P,) scores (roll out once); pass 2 re-rolls out just the ten render episodes (needed
for per-step frames, which pass 1 does not keep) and applies the now-known PC1 mask.
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
PC1_PCT = 75
EPISODES = ["50", "54", "59", "61", "65", "78", "79", "82", "51", "52"]


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/sanger/wksp/panda_express/tasks/jenga_tilt_100/jenga_tilt.lmdb")
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--outdir", default="/home/sanger/wksp/panda_express/results/pc1_videos")
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

        def roll(ep, s, keys, acts, props, want_frames):
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
            de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).mean(0).cpu().numpy()  # (P,)
            centroid = zn[:, -1].mean(0, keepdim=True)
            fv = (1 - cs(zn[:, -1], centroid, dim=-1)).mean(0).cpu().numpy()            # (P,)
            z_obs = zv[0, NH - 1].cpu().numpy().astype(np.float32)                      # (P,F)
            norm = zv[0, NH - 1].norm(dim=-1).cpu().numpy()
            return de, fv, z_obs, norm, (imgs if want_frames else None)

        # ---- pass 1: build the PC1 fitting bank + safe-chunk scores, all held-out episodes ----
        print(f"pass 1: {len(held)} held-out episodes", flush=True)
        safe_de, safe_fv, safe_zobs, safe_norm = [], [], [], []
        for i, ep in enumerate(held):
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                if fstep is not None and lo <= fstep <= hi:
                    continue                        # unsafe: excluded from PC1 fit/calibration
                r = roll(ep, s, keys, acts, props, False)
                if r is None:
                    break
                de, fv, z_obs, norm, _ = r
                safe_de.append(de); safe_fv.append(fv); safe_zobs.append(z_obs); safe_norm.append(norm)
            if i % 10 == 0:
                print(f"  [{i}/{len(held)}] safe chunks={len(safe_de)}", flush=True)

        bank = np.concatenate([z[keep] for z in safe_zobs], 0)
        mu = bank.mean(0)
        _, S, Vt = np.linalg.svd(bank - mu, full_matrices=False)
        pc1_all = np.concatenate([((z - mu) @ Vt[0])[keep] for z in safe_zobs])
        nrm_all = np.concatenate([nn[keep] for nn in safe_norm])
        fg_sign = 1 if nrm_all[pc1_all > np.median(pc1_all)].mean() > \
                       nrm_all[pc1_all <= np.median(pc1_all)].mean() else -1
        print(f"PC1 fit on {bank.shape[0]} safe patch vectors, top1 var "
              f"{(S**2/(S**2).sum())[0]:.3f}, foreground sign {'+' if fg_sign>0 else '-'}")

        def pc1_mask(z_obs):
            pc1 = ((z_obs - mu) @ Vt[0]) * fg_sign
            m = keep.copy()
            m &= pc1 >= np.percentile(pc1[keep], 100 - PC1_PCT)
            return m

        def score(v, mask):
            x = v[mask]; x = x[np.isfinite(x)]
            return float(np.percentile(x, 90)) if x.size >= 4 else np.nan

        safe_de_pc1 = [score(de, pc1_mask(z)) for de, z in zip(safe_de, safe_zobs)]
        safe_fv_pc1 = [score(fv, pc1_mask(z)) for fv, z in zip(safe_fv, safe_zobs)]
        THR = {"d_end_pc1": float(np.nanpercentile(safe_de_pc1, 90)),
               "ftle_variance_pc1": float(np.nanpercentile(safe_fv_pc1, 90))}
        print(f"thresholds (p90 of {len(safe_de)} safe chunks): "
              f"d_end_pc1={THR['d_end_pc1']:.4f}  "
              f"ftle_variance_pc1={THR['ftle_variance_pc1']:.4f}")

        # ---- pass 2: re-roll the 10 render episodes (need per-step frames), apply PC1 mask ----
        for ep in EPISODES:
            if ep not in meta["episodes"]:
                continue
            keys, acts, props, tilt, n = load(ep)
            over = np.where(tilt.max(axis=1) > TOPPLE)[0]
            fstep = int(over[0]) if len(over) else None
            starts, v_de, v_fv, gts, frames = [], [], [], [], {}
            for s in range(0, n - span, NP):
                lo, hi = s + NH, s + span - 1
                if fstep is not None and fstep < lo:
                    break
                r = roll(ep, s, keys, acts, props, True)
                if r is None:
                    break
                de, fv, z_obs, norm, imgs = r
                m = pc1_mask(z_obs)
                starts.append(s); v_de.append(score(de, m)); v_fv.append(score(fv, m))
                frames[s] = imgs
                gts.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
            if not starts:
                continue

            for tag, vals, colour in (("d_end_pc1", v_de, "#1565c0"),
                                      ("ftle_variance_pc1", v_fv, "#2e7d32")):
                thr = THR[tag]
                fire = next((s for s, v in zip(starts, vals) if np.isfinite(v) and v > thr), None)
                lead = (fstep - fire) if (fire is not None and fstep is not None) else None
                failed = fstep is not None
                status = (("FALSEALARM" if fire is not None else "ok-silent") if not failed
                          else ("CAUGHT" if fire is not None else "MISSED"))
                vid = out / f"ep{ep}_{tag}_{status}.mp4"
                fvals = [v for v in vals if np.isfinite(v)]
                lo_y = min(fvals + [thr]); hi_y = max(fvals + [thr])
                pad = 0.15 * (hi_y - lo_y + 1e-6)
                vw, H = None, 480
                for gi, s in enumerate(starts):
                    fired = fire is not None and s >= fire
                    for im in frames[s][NH:]:
                        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
                        ax.plot(starts[:gi + 1], vals[:gi + 1], "s-", color=colour, lw=2.5,
                                ms=4, label=f"{tag} (p90, PC1 75% mask)")
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
                        ax2.set_ylim(-5, 100)
                        ax2.set_ylabel("ground-truth tilt (deg)", color="#555", fontsize=9)
                        ax2.axhline(TOPPLE, color="#888", ls=":", lw=1)
                        ttl = f"ep {ep} · {tag}"
                        if lead is not None:
                            ttl += f" · HALT {lead} steps early" if lead >= 0 else " · HALT (late)"
                        elif failed:
                            ttl += " · NO ALARM"
                        else:
                            ttl += " · safe run"
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
