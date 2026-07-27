"""Verify the fused-attention swap in models/vit.py is numerically equivalent.

Runs the identical rollout twice on identical inputs -- once through
F.scaled_dot_product_attention, once through the original explicit-matmul path -- and
compares the resulting latents and FTLE. Any optimisation that changes the score is not
an optimisation, so this has to pass before the speedup is usable.
"""
import pickle
from pathlib import Path

import cv2, hydra, lmdb, numpy as np, torch
from einops import rearrange
from torchvision import transforms

from server_single_max import load_model
import models.vit as vitmod

AM = [0.45678952, 0.00051019, 0.50954217, 0.21926114]
ASD = [0.03182372, 0.01151787, 0.03419121, 0.41397065]
PM = [0.4564166, 0.00056233, 0.50817657, 0.21921302]
PSD = [0.03217997, 0.01056713, 0.0327194, 0.4139551]
NH, NP, N, MASK = 3, 8, 50, 28
dev = "cuda"


def manual_forward(self, x):
    """The original explicit-matmul attention, restored verbatim."""
    B, T, C = x.size()
    x = self.norm(x)
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    dots = dots.masked_fill(self.bias.to(dots.device)[:, :, :T, :T] == 0, float("-inf"))
    attn = self.dropout(self.attend(dots))
    out = torch.matmul(attn, v)
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


def ftle(zv):
    zo, zn = zv[0:1], zv[1:]
    ds = (1 - torch.nn.functional.cosine_similarity(zn[:, NH], zo[:, NH], dim=-1)) + 1e-4
    de = (1 - torch.nn.functional.cosine_similarity(zn[:, -1], zo[:, -1], dim=-1)) + 1e-4
    lam = (1.0 / NP) * torch.log(de / ds)
    lam[:, :MASK] = -float("inf")
    lam[de < 1e-3] = -float("inf")
    return lam.max().item(), lam


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

torch.manual_seed(1234)                      # identical perturbations for both runs
a = torch.from_numpy(acts[s:s + span]).float().to(dev).unsqueeze(0).repeat(N, 1, 1)
a[1:, :, :3] += torch.randn(N - 1, span, 3, device=dev) * 0.05
a = (a - am) / asd

with torch.no_grad():
    z_fast, _ = model.rollout(obs, a)

orig = vitmod.Attention.forward
vitmod.Attention.forward = manual_forward
with torch.no_grad():
    z_ref, _ = model.rollout(obs, a)
vitmod.Attention.forward = orig

zf, zr = z_fast["visual"], z_ref["visual"]
f_fast, lam_fast = ftle(zf)
f_ref, lam_ref = ftle(zr)
d = (zf - zr).abs()

print(f"latent max abs diff   : {d.max().item():.3e}")
print(f"latent mean abs diff  : {d.mean().item():.3e}   (latent scale ~{zr.abs().mean().item():.3f})")
print(f"FTLE fused            : {f_fast:.10f}")
print(f"FTLE reference        : {f_ref:.10f}")
print(f"FTLE abs diff         : {abs(f_fast - f_ref):.3e}")
fin = torch.isfinite(lam_fast) & torch.isfinite(lam_ref)
print(f"per-perturbation FTLE max diff : {(lam_fast[fin]-lam_ref[fin]).abs().max().item():.3e}")
print(f"argmax patch identical         : {lam_fast.argmax().item() == lam_ref.argmax().item()}")
