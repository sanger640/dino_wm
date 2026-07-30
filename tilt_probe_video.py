"""
Render what the tilt probe sees, next to what the divergence metric sees.

The probe numbers (predictor R^2 0.836, AUC 0.992 vs d_end's corr 0.038) say the world
model's predicted latent knows the block is tipping and the divergence readout does not.
This makes that visible per frame: ground-truth tilt, the probe's prediction of tilt 8
steps ahead, and d_end, all on the same timeline.

Two efficiency notes that matter for how long this takes:

  * FITTING the probe needs only the unperturbed rollout, so it runs at N=1 (~53 ms/chunk)
    rather than N=50 (~2 s). The probe never looks at the perturbations.
  * Only the RENDERED episodes pay for N=50, and only so d_end can be drawn alongside.

The probe is fit on one half of the episodes and every rendered episode comes from the
OTHER half, so nothing shown was trained on.
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
    ap.add_argument("--n-videos", type=int, default=5)
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--outdir", default="/home/sanger/wksp/panda_express/results/tilt_videos")
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
    cs = torch.nn.functional.cosine_similarity
    env = lmdb.open(args.lmdb, readonly=True, lock=False)

    def rollout(txn, ep, s, keys, acts, props, N):
        raw = [txn.get(keys[s + j].encode()) for j in range(span)]
        if any(r is None for r in raw):
            return None, None
        imgs = [dec(r) for r in raw]
        vis = tf(torch.from_numpy(np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])
                                  ).float().to(dev) / 255.)
        a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
        if N > 1:
            g = torch.Generator(device=dev); g.manual_seed(s)
            a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * args.noise_std
        obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
               "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd
                           ).unsqueeze(0).repeat(N, 1, 1)}
        with torch.no_grad():
            z, _ = model.rollout(obs, (a - am) / asd)
        return z["visual"], imgs

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = sorted(meta["episodes"], key=lambda e: int(e))
        fit_eps, held_eps = eps[:args.n_fit_episodes], eps[args.n_fit_episodes:]

        # ---------- fit the probe (N=1: the probe never uses perturbations) ----------
        print(f"fitting probe on {len(fit_eps)} episodes at N=1", flush=True)
        X, Y = [], []
        for i, ep in enumerate(fit_eps):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            n = min(len(keys), len(acts), len(props), len(tilt))
            for s in range(0, n - span, NP):
                zv, _ = rollout(txn, ep, s, keys, acts, props, 1)
                if zv is None:
                    break
                X.append(zv[0, -1][keep].mean(0).cpu().numpy())
                Y.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
            if i % 10 == 0:
                print(f"  [{i}/{len(fit_eps)}] n={len(X)}", flush=True)
        X = np.stack(X); Y = np.array(Y)
        w = ridge_fit(X, Y, args.lam)
        print(f"probe fit on {len(X)} chunks; train R2 "
              f"{1 - ((Y - ridge_apply(X, w))**2).sum() / ((Y - Y.mean())**2).sum():.3f}")

        # ---------- choose held-out episodes: mix of failures and successes ----------
        fails, succs = [], []
        for ep in held_eps:
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            (fails if tilt.max() > TOPPLE else succs).append(ep)
        nf = min(len(fails), max(1, args.n_videos - 2))
        chosen = fails[:nf] + succs[:args.n_videos - nf]
        print(f"held-out: {len(fails)} failures, {len(succs)} successes -> rendering {chosen}")

        for ep in chosen:
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            tilt = pickle.loads(txn.get(f"{ep}_tilt".encode()))
            n = min(len(keys), len(acts), len(props), len(tilt))
            starts, pred_t, gt_t, dend = [], [], [], []
            frames_by_start = {}
            for s in range(0, n - span, NP):
                zv, imgs = rollout(txn, ep, s, keys, acts, props, args.n_perturb)
                if zv is None:
                    break
                zo, zn = zv[0:1], zv[1:]
                de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).mean(0).cpu().numpy()
                starts.append(s)
                pred_t.append(float(ridge_apply(zv[0, -1][keep].mean(0).cpu().numpy()[None], w)[0]))
                gt_t.append(float(tilt[min(s + span - 1, len(tilt) - 1)].max()))
                dend.append(float(np.percentile(de[keep], 90)))
                frames_by_start[s] = imgs
            if not starts:
                continue
            failed = max(gt_t) > TOPPLE
            vid = out / f"ep{ep}_{'FAIL' if failed else 'ok'}.mp4"
            vw, H = None, 480
            for gi, s in enumerate(starts):
                imgs = frames_by_start[s]
                for fi, im in enumerate(imgs[NH:], start=NH):
                    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
                    ax.plot(starts[:gi + 1], gt_t[:gi + 1], "o-", color="#111",
                            lw=2.5, ms=4, label="ground-truth tilt (t+8)")
                    ax.plot(starts[:gi + 1], pred_t[:gi + 1], "s--", color="#c62828",
                            lw=2.5, ms=4, label="PROBE prediction")
                    ax.axhline(TOPPLE, color="#888", ls=":", lw=1.5)
                    ax.text(0, TOPPLE + 2, "topple = 45 deg", color="#666", fontsize=8)
                    ax.set_xlim(0, max(starts)); ax.set_ylim(-5, 100)
                    ax.set_xlabel("rollout step"); ax.set_ylabel("tilt (deg)")
                    ax2 = ax.twinx()
                    ax2.plot(starts[:gi + 1], dend[:gi + 1], "^-", color="#1565c0",
                             lw=1.8, ms=3, alpha=.75, label="d_end p90 (current metric)")
                    ax2.set_ylim(0, max(max(dend) * 1.3, 0.05)); ax2.set_ylabel("d_end", color="#1565c0")
                    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
                    ax.set_title(f"ep {ep} — {'TOPPLE' if failed else 'success'}"
                                 f"   GT {gt_t[gi]:.1f}°   probe {pred_t[gi]:.1f}°", fontsize=10)
                    fig.tight_layout(); fig.canvas.draw()
                    plot = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                    plt.close(fig)

                    left = cv2.resize(im, (int(H * im.shape[1] / im.shape[0]), H))
                    plot = cv2.resize(plot, (int(H * plot.shape[1] / plot.shape[0]), H))
                    canvas = np.hstack([left, plot])
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                    if vw is None:
                        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (canvas.shape[1], canvas.shape[0]))
                    vw.write(canvas)
            if vw is not None:
                vw.release()
            err = np.abs(np.array(pred_t) - np.array(gt_t)).mean()
            print(f"  {vid.name}: {len(starts)} chunks, mean |probe-GT| = {err:.1f} deg, "
                  f"peak GT {max(gt_t):.1f} deg", flush=True)
    env.close()
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
