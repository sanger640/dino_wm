"""
Do the perturbations perturb anything the predictor reacts to?

If the Deviator Agent works, d_end must grow with the perturbation magnitude sigma: bigger
action deviations should produce more divergent predicted futures. If d_end is flat in
sigma, the ViT predictor is effectively ignoring the action noise, and d_end is measuring
scene change alone -- which would mechanically explain why a single unperturbed rollout
matched the full N=50 apparatus.

Reported per sigma:
    d_end        absolute level, safe chunks vs unsafe chunks
    slope        d(d_end)/d(sigma) -- flat means the perturbations are inert
    AUC          discrimination, to see whether any sigma is better than the default 0.05

Runs on a subset (every unsafe chunk plus a capped sample of safe ones) because the sweep
multiplies cost by the number of sigmas. Includes sigma=0 as a control: at sigma=0 all 50
rollouts are identical, so d_end must be ~0 and AUC ~chance. If it is not, the harness is
wrong, not the model.
"""
import argparse, json, pickle
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
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]


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
    ap.add_argument("--n-safe", type=int, default=200, help="safe chunks sampled")
    ap.add_argument("--out", default="outputs/sigma_sweep.json")
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
    rng = np.random.default_rng(0)

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
        print(f"{len(targets)} chunks ({len(pos)} unsafe, {len(neg)} safe) x {len(SIGMAS)} sigmas")

        out = {sg: {"y": [], "dend": [], "nominal": []} for sg in SIGMAS}
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
            pro = ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd)
            a0 = torch.from_numpy(acts[s:s + span]).float().to(dev)
            obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
                   "proprio": pro.unsqueeze(0).repeat(N, 1, 1)}
            for sg in SIGMAS:
                g = torch.Generator(device=dev); g.manual_seed(s)
                a = a0.unsqueeze(0).repeat(N, 1, 1)
                if sg > 0:
                    a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev, generator=g) * sg
                with torch.no_grad():
                    z, _ = model.rollout(obs, (a - am) / asd)
                zv = z["visual"]; zo, zn = zv[0:1], zv[1:]
                de = ((1 - cs(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4).mean(0).cpu().numpy()
                nm = (1 - cs(zo[0, NH], zo[0, -1], dim=-1)).cpu().numpy()
                out[sg]["y"].append(y)
                out[sg]["dend"].append(float(np.percentile(de[keep], 90)))
                out[sg]["nominal"].append(float(np.percentile(nm[keep], 90)))
            if i % 25 == 0:
                print(f"  [{i}/{len(targets)}]", flush=True)
    env.close()

    print(f"\n{'sigma':>7}{'d_end safe':>12}{'d_end unsafe':>14}{'ratio':>8}"
          f"{'AUC d_end':>11}{'AUC nominal':>13}")
    print("-" * 65)
    lvl = []
    for sg in SIGMAS:
        y = np.array(out[sg]["y"]); d = np.array(out[sg]["dend"]); nm = np.array(out[sg]["nominal"])
        sa, un = d[y == 0].mean(), d[y == 1].mean()
        lvl.append(d.mean())
        print(f"{sg:>7.3f}{sa:>12.4f}{un:>14.4f}{un/max(sa,1e-9):>8.2f}"
              f"{auc(d[y==1], d[y==0]):>11.3f}{auc(nm[y==1], nm[y==0]):>13.3f}")

    nz = [s for s in SIGMAS if s > 0]
    lz = [lvl[SIGMAS.index(s)] for s in nz]
    sl = np.polyfit(np.log(nz), np.log(lz), 1)[0]
    print(f"\nlog-log slope of d_end vs sigma (sigma>0): {sl:+.3f}")
    print("  ~0    => predictor ignores the action noise; d_end is scene change alone")
    print("  ~1    => divergence scales linearly with the perturbation, as intended")
    print(f"\nd_end at sigma=0 (control, must be ~0): {lvl[0]:.6f}")
    print("nominal AUC is sigma-independent by construction -- it uses only the unperturbed"
          " rollout, so any variation across rows is sampling noise, and it is the"
          " reference line the d_end column has to beat.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({str(k): v for k, v in out.items()}, open(args.out, "w"))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
