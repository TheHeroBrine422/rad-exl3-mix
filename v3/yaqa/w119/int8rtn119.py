#!/usr/bin/env python3
"""int8rtn119.py CKPT OUT.npz [fwht] -- w119 E9 (RULES.md amendment 1): the lm_head as radiance's `rtn codes=i8 group=128
scale=bf16 clamp=sym rule=search` would store it (lq_grid.cpp: absmax scale s0 = absmax / 127; search_sym over
s0 x (1 + 0.02 k), k = -15..5, and each candidate's least-squares refit sxd / sdd, every candidate rounded to bf16 before
it is judged, no importance weights; codes round half away from zero, clamped to +-127). fwht: the rows' 128-input
groups rotated by the normalised Sylvester Hadamard first (report-only arm). OUT.npz: codes int8 [V, 5120], scale f32
[V, 40] (bf16 values)."""
import json, sys
import numpy as np, torch
from safetensors import safe_open

CK, OUT = sys.argv[1:3]; FWHT = len(sys.argv) > 3 and sys.argv[3] == "fwht"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: W = f.get_tensor("lm_head.weight")
V, C = W.shape; G = C // 128
h = torch.tensor([[1.0]], dtype=torch.float64)
while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
HAD = (h / 128 ** 0.5).to(dev)
codes = np.zeros((V, C), np.int8); scale = np.zeros((V, G), np.float32)


def bf(s): return s.float().to(torch.bfloat16).double()


def rnd(q): return torch.where(q < 0, torch.ceil(q - 0.5), torch.floor(q + 0.5)).clamp(-127, 127)


def fit(x, s):                                       # x [r, G, 128], s [r, G, 1] -> err, sxd, sdd, codes
    c = rnd(x / s); e = x - s * c
    return (e * e).sum(-1), (x * c).sum(-1), (c * c).sum(-1), c


R = 4096; num = den = 0.0
with torch.no_grad():
    for r0 in range(0, V, R):
        x = W[r0:r0 + R].to(dev).double().view(-1, G, 128)
        if FWHT: x = x @ HAD
        s0 = x.abs().amax(-1) / 127
        nz = s0 > 0
        s0 = torch.where(nz, s0, torch.ones_like(s0))
        best = bf(s0); be, _, _, _ = fit(x, best[..., None])
        for k in range(-15, 6):
            s = bf(s0 * (1 + 0.02 * k)); e, sxd, sdd, _ = fit(x, s[..., None])
            m = e < be; best = torch.where(m, s, best); be = torch.where(m, e, be)
            ok = (sdd > 0) & (sxd != 0)
            t = bf(torch.where(ok, sxd / torch.where(ok, sdd, torch.ones_like(sdd)), s))
            e2, _, _, _ = fit(x, t[..., None])
            m = ok & (e2 < be); best = torch.where(m, t, best); be = torch.where(m, e2, be)
        best = torch.where(nz, best, torch.zeros_like(best))
        c = torch.where(nz[..., None], rnd(x / torch.where(nz, best, torch.ones_like(best))[..., None]), torch.zeros_like(x))
        codes[r0:r0 + R] = c.view(-1, C).to(torch.int8).cpu().numpy(); scale[r0:r0 + R] = best.float().cpu().numpy()
        num += float(((c * best.float().double()[..., None] - x) ** 2).sum()); den += float((x * x).sum())
rel = num / den
np.savez(OUT, codes=codes, scale=scale)
print(f"saved {OUT} ({'fwht128 ' if FWHT else ''}rtn search): relative squared error {rel:.3e}")
