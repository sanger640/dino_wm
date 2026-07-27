"""Profile where time goes in a single safety-monitor rollout (N=50, T=8)."""
import time, pickle
from pathlib import Path
import cv2, hydra, lmdb, numpy as np, torch
from torchvision import transforms
from server_single_max import load_model

AM = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ASD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PM = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PSD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]
NH, NP, N = 3, 8, 50
dev = "cuda"

with hydra.initialize(config_path="conf", version_base=None):
    cfg = hydra.compose(config_name="train")
model = load_model(Path("outputs/model_latest_single.pth"), cfg, dev)
model.eval()
tf = transforms.Compose([transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])

env = lmdb.open("/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single.lmdb",
                readonly=True, lock=False)
with env.begin() as t:
    m = pickle.loads(t.get(b"__metadata__")); ep = "8"
    keys = m["episodes"][ep]["keys"]["cam2"]
    acts = pickle.loads(t.get(f"{ep}_actions".encode()))
    props = pickle.loads(t.get(f"{ep}_proprio".encode()))
    s, span = 86, NH + NP
    fr = [cv2.cvtColor(cv2.imdecode(np.frombuffer(t.get(keys[s + i].encode()), np.uint8), 1),
                       cv2.COLOR_BGR2RGB) for i in range(span)]

vis = torch.from_numpy(np.stack([np.transpose(f, (2, 0, 1)) for f in fr])).float().to(dev) / 255.
vis = tf(vis)
am = torch.tensor(AM, device=dev); asd = torch.tensor(ASD, device=dev)
pm = torch.tensor(PM, device=dev); psd = torch.tensor(PSD, device=dev)
obs = {"visual": vis[:NH].unsqueeze(0).repeat(N, 1, 1, 1, 1),
       "proprio": ((torch.from_numpy(props[s:s + NH]).float().to(dev) - pm) / psd).unsqueeze(0).repeat(N, 1, 1)}
a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev) * 0.05
a = (a - am) / asd


def tm(fn, n=5):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        r = fn()
    torch.cuda.synchronize(); return (time.time() - t0) / n, r


with torch.no_grad():
    t_enc, _ = tm(lambda: model.encode_obs(obs))
    t_full, (zo, _) = tm(lambda: model.rollout(obs, a), 3)
    zfull = model.encode(obs, a[:, :NH])
    t_p1, _ = tm(lambda: model.predict(zfull[:, -NH:]))
    t_dec, _ = tm(lambda: model.decode_obs({k: v[[0, 1]] for k, v in zo.items()}), 3)

print(f"  encode_obs (deduped)    : {t_enc*1000:8.1f} ms")
print(f"  predict() one step      : {t_p1*1000:8.1f} ms  x{NP+1} = {t_p1*(NP+1)*1000:8.0f} ms")
print(f"  decode_obs (2 traj)     : {t_dec*1000:8.1f} ms")
print(f"  full rollout()          : {t_full*1000:8.1f} ms")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    t_half, _ = tm(lambda: model.rollout(obs, a), 3)
print(f"  rollout() autocast fp16 : {t_half*1000:8.1f} ms   -> {t_full/t_half:.2f}x")

# does fp16 change the score?
def ftle(zv):
    zo_, zn_ = zv[0:1], zv[1:]
    ds = (1 - torch.nn.functional.cosine_similarity(zn_[:, NH], zo_[:, NH], dim=-1)) + 1e-4
    de = (1 - torch.nn.functional.cosine_similarity(zn_[:, -1], zo_[:, -1], dim=-1)) + 1e-4
    lam = (1.0 / NP) * torch.log(de / ds)
    lam[:, :28] = -float("inf"); lam[de < 1e-3] = -float("inf")
    return float(lam.max())


with torch.no_grad():
    z32, _ = model.rollout(obs, a)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    z16, _ = model.rollout(obs, a)
f32, f16 = ftle(z32["visual"].float()), ftle(z16["visual"].float())
print(f"\n  FTLE fp32 {f32:.6f}   fp16 {f16:.6f}   delta {abs(f32-f16):.2e}")
