"""
Exact FTLE via the flow-map Jacobian, instead of a max over random samples.

The sampled estimator was measured to be poor: N=50 recovers only 80% of the maximum found
at N=400 (which itself has not plateaued), with a 17% coefficient of variation across
disjoint blocks of 50. That noise goes straight into every score. The perturbation lives in
span x 3 = 33 dimensions, where a random direction has expected alignment ~1/sqrt(33) = 0.17
with the top singular vector, so under-coverage is expected rather than surprising.

The exact quantity:

    J = d z_T / d a                      (patches x 384 x 33)
    lambda = (1/T) * ln sigma_max(J)     sigma_max from the largest eigenvalue of J^T J

Two details that matter for correctness:

  TANGENT SPACE. The score is a cosine distance, which depends only on the DIRECTION of
  z_T. A raw Jacobian mixes radial growth (irrelevant -- cosine is scale invariant) with
  angular growth (what is actually measured). So J is projected onto the tangent space at
  z_T and divided by ||z_T||:  Jhat = (I - zhat zhat^T) J / ||z||. Skipping this conflates
  the two and inflates sigma_max on exactly the high-norm patches.

  IT IS NOT A FLOW-MAP JACOBIAN. Classical FTLE differentiates state w.r.t. INITIAL STATE.
  Here the perturbation is applied to ACTIONS, so this is an input-output sensitivity of the
  same composed 8-step map. Worth stating plainly in the paper rather than calling it a
  Lyapunov exponent without qualification.

Both variants are computed, since the sampled version showed the denominator hurts:
    jac_sigma_end   sigma_max of d z_T / d a          (analogue of d_end)
    jac_ftle        (1/T) ln(sigma_end / sigma_start)  (analogue of the full FTLE)

Correctness check built in: sigma_max should correlate strongly with the sampled max
divergence across chunks. If it does not, the Jacobian is wrong, and the script says so
before any AUC is reported.
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
LOWNORM_K = 30


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


def sigma_max_tangent(J, z):
    """Largest singular value per patch, in the tangent space at z.

    J: (P, F, D)   z: (P, F)  ->  (P,)
    Cosine distance is scale-invariant, so only the component of the perturbation
    orthogonal to z changes the score; the radial part must be projected out.
    """
    nrm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)          # (P,1)
    zhat = z / nrm
    radial = torch.einsum("pf,pfd->pd", zhat, J)                 # (P,D)
    Jt = (J - zhat.unsqueeze(-1) * radial.unsqueeze(1)) / nrm.unsqueeze(-1)
    M = torch.einsum("pfd,pfe->pde", Jt, Jt)                     # (P,D,D)
    ev = torch.linalg.eigvalsh(M.double())
    return ev[:, -1].clamp_min(0).sqrt().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-safe", type=int, default=200)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--out", default="outputs/jacobian_ftle.json")
    args = ap.parse_args()
    N = args.n_perturb

    with hydra.initialize(config_path="conf", version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_model(Path("outputs/model_latest_single.pth"), cfg, dev); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
    pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)
    labels = json.load(open(LABELS))
    keep = build_patch_keep_mask(196, torch.device("cpu")).numpy()
    span = NH + NP
    D = span * 3
    cs = torch.nn.functional.cosine_similarity
    rng = np.random.default_rng(0)
    env = lmdb.open(LMDB, readonly=True, lock=False)
    print(f"perturbation dimension D = span({span}) x 3 = {D}")

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        pos, neg = [], []
        for ep, v in labels.items():
            if ep not in meta["episodes"]:
                continue
            keys = meta["episodes"][ep]["keys"]["cam2"]
            f = v["failure_step"] if v["outcome"] == "failure" else None
            for s in range(0, len(keys) - span, NP):
                lo, hi = s + NH, s + span - 1
                if f is not None and f < lo:
                    break
                (pos if (f is not None and f <= hi) else neg).append((ep, s))
        neg = [neg[i] for i in rng.choice(len(neg), min(args.n_safe, len(neg)), replace=False)]
        targets = [(e, s, 1) for e, s in pos] + [(e, s, 0) for e, s in neg]
        if args.max_chunks:
            # interleave so a truncated smoke run still contains both classes
            rng.shuffle(targets)
            targets = targets[:args.max_chunks]
        print(f"{len(targets)} chunks ({sum(t[2] for t in targets)} unsafe)", flush=True)

        rows, t_jac, t_smp, mode = [], 0.0, 0.0, None
        for i, (ep, s, y) in enumerate(targets):
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            if s + span > min(len(acts), len(props)):
                continue
            raw = [txn.get(keys[s + j].encode()) for j in range(span)]
            if any(r is None for r in raw):
                continue
            vis = tf(torch.from_numpy(np.stack([np.transpose(dec(r), (2, 0, 1)) for r in raw])
                                      ).float().to(dev) / 255.)
            a0 = torch.from_numpy(acts[s:s + span]).float().to(dev)
            pro = ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd)
            obs1 = {"visual": vis[:NH].unsqueeze(0), "proprio": pro.unsqueeze(0)}

            def f(delta):
                """delta (D,) raw action offset on xyz -> [z_NH, z_T] (2,P,F)."""
                a = a0 + torch.cat([delta.view(span, 3),
                                    torch.zeros(span, 1, device=dev, dtype=delta.dtype)], -1)
                z, _ = model.rollout(obs1, ((a - am) / asd).unsqueeze(0))
                return torch.stack([z["visual"][0, NH], z["visual"][0, -1]])

            zero = torch.zeros(D, device=dev)
            torch.cuda.synchronize(); t0 = time.time()
            if mode != "fd":
                try:
                    # The efficient/flash SDPA kernels do not implement forward-mode AD
                    # (they were adopted earlier for a 1.6x speedup). The MATH backend does,
                    # and it is only needed for this differentiated pass.
                    from torch.func import jacfwd
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    with sdpa_kernel([SDPBackend.MATH]):
                        J = jacfwd(f)(zero)                   # (2,P,F,D)
                    mode = "jacfwd"
                except Exception as e:                        # noqa: BLE001
                    if mode is None:
                        print(f"  forward-mode AD unavailable ({type(e).__name__}: "
                              f"{str(e)[:90]}) -> exact basis finite differences", flush=True)
                    mode = "fd"
            if mode == "fd":
                # Deterministic orthogonal basis: D one-sided differences give the same
                # Jacobian to first order, without relying on forward-mode support.
                eps = 1e-3
                base = f(zero)
                cols = []
                for d in range(D):
                    e = torch.zeros(D, device=dev); e[d] = eps
                    cols.append((f(e) - base) / eps)
                J = torch.stack(cols, dim=-1)
            torch.cuda.synchronize(); t_jac += time.time() - t0

            with torch.no_grad():
                z_ref = f(zero)
                sig_s = sigma_max_tangent(J[0], z_ref[0])
                sig_e = sigma_max_tangent(J[1], z_ref[1])

                # sampled baseline on the same chunk
                torch.cuda.synchronize(); t1 = time.time()
                g = torch.Generator(device=dev); g.manual_seed(s)
                a = a0.unsqueeze(0).repeat(N, 1, 1)
                a[:, :, :3] += torch.randn(N, span, 3, device=dev, generator=g) * args.noise_std
                obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                       "proprio": pro.unsqueeze(0).repeat(N, 1, 1)}
                zs, _ = model.rollout(obs, (a - am) / asd)
                torch.cuda.synchronize(); t_smp += time.time() - t1
                de = ((1 - cs(zs["visual"][:, -1], z_ref[1].unsqueeze(0), dim=-1)) + 1e-4)
                # per-patch max over perturbations -- the sampled analogue of sigma_max,
                # so the correctness check compares like with like (both p90 over patches)
                de_patchmax = de.max(dim=0).values.cpu().numpy()
                nrm = z_ref[0].norm(dim=-1).cpu().numpy()

            m = keep.copy(); m[np.argsort(nrm)[:LOWNORM_K]] = False
            se = sig_e.cpu().numpy(); ss = sig_s.cpu().numpy()
            den = np.where(ss > 1e-9, ss, np.nan)
            rows.append({
                "ep": ep, "y": y,
                "jac_sigma_end": float(np.percentile(se[m], 90)),
                "jac_sigma_end_max": float(se[m].max()),
                "jac_ftle": float(np.nanpercentile((1.0 / NP) * np.log(se / den)[m], 90)),
                "dend_p90": float(np.percentile(de[:, m].mean(0).cpu().numpy(), 90)),
                "dend_maxpert": float(np.percentile(de_patchmax[m], 90)),
                "sig_e_mean": float(se[m].mean()),
            })
            if i % 25 == 0:
                print(f"  [{i}/{len(targets)}] mode={mode}", flush=True)
    env.close()

    n = len(rows)
    ys = np.array([r["y"] for r in rows])
    print(f"\n{n} chunks | {int(ys.sum())} unsafe | {int((1-ys).sum())} safe")
    print(f"latency/chunk: jacobian({mode}) {1000*t_jac/n:.0f} ms   "
          f"sampled N={N} {1000*t_smp/n:.0f} ms")

    # ---------- correctness: sigma_max must track the sampled divergence ----------
    a1 = np.array([r["jac_sigma_end"] for r in rows])
    b1 = np.array([r["dend_maxpert"] for r in rows])   # both are p90-over-patches
    ok = np.isfinite(a1) & np.isfinite(b1)
    c = np.corrcoef(a1[ok], b1[ok])[0, 1]
    print(f"\n=== CORRECTNESS CHECK ===")
    print(f"  corr(sigma_max, sampled max divergence) = {c:+.3f}")
    print("  expect clearly positive and strong; near zero would mean the Jacobian is wrong")
    if not (c > 0.5):
        print("  ** WARNING: weak correlation -- treat the AUCs below as unreliable **")

    print("\n=== AUC (unsafe vs safe), same chunks ===")
    res = {}
    for k in ("jac_sigma_end", "jac_sigma_end_max", "jac_ftle", "sig_e_mean",
              "dend_p90", "dend_maxpert"):
        v = np.array([r[k] for r in rows])
        res[k] = auc(v[ys == 1], v[ys == 0])
        print(f"  {k:<22}{res[k]:.3f}")
    print("\n  jac_* = exact Jacobian; dend_* = sampled N=50 baseline on identical chunks")

    # paired bootstrap vs the sampled baseline
    eps_u = sorted({r["ep"] for r in rows})
    idx = {e: [i for i, r in enumerate(rows) if r["ep"] == e] for e in eps_u}
    rng2 = np.random.default_rng(0)
    va = np.array([r["jac_sigma_end"] for r in rows])
    vb = np.array([r["dend_p90"] for r in rows])
    ds = []
    for _ in range(2000):
        sel = np.concatenate([idx[e] for e in rng2.choice(eps_u, len(eps_u), replace=True)])
        yy = ys[sel]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        ds.append(auc(va[sel][yy == 1], va[sel][yy == 0])
                  - auc(vb[sel][yy == 1], vb[sel][yy == 0]))
    ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"\njac_sigma_end vs dend_p90: {res['jac_sigma_end']:.3f} vs {res['dend_p90']:.3f}"
          f"  diff {res['jac_sigma_end']-res['dend_p90']:+.3f}"
          f"  95% CI [{lo:+.3f}, {hi:+.3f}]  P(A>B)={(ds>0).mean():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": n, "mode": mode, "corr_check": float(c), "auc": res,
               "ms_jacobian": 1000 * t_jac / n, "ms_sampled": 1000 * t_smp / n},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
