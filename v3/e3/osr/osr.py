#!/usr/bin/env python3
"""osr.py LAYER... -- upstream exllamav3's output-scale test (quantize.py out_scales_ratio, auto mode drops the scales
above 1.05) for gate / up of each layer: ratio = E[s e] / (E[s] E[e]), e = output channel energy (row mean square),
s = mean square slope of act(g) * u along the projection x down column energy (mlp.py capture_out_sensitivity).
Inputs x ~ N(0, H / rows) from the published calib Hessian (the second moment: H is a raw sum over `rows` tokens).
Also prints the ratio with the raw sum (what a first review estimate used), to show how much the scale matters.
Env E3_CKPT, E3_HESS, OSR_N (samples, 8192). One JSON line per layer."""
import json, math, os, sys
import torch
from safetensors import safe_open
torch.manual_seed(0)
CK = os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16"); HB = os.environ.get("E3_HESS", "/data/challenge/hessians/calib")
N = int(os.environ.get("OSR_N", 8192))
ix = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
def T(k):
    with safe_open(f"{CK}/{ix[k]}", "pt") as f: return f.get_tensor(k).float()
def H(fn):
    with safe_open(f"{HB}/{fn}", "pt") as f: hu = f.get_tensor("hessian_upper").double(); md = f.metadata()
    n = int((math.isqrt(8 * hu.numel() + 1) - 1) // 2); Hm = torch.zeros(n, n, dtype=torch.float64)
    iu = torch.triu_indices(n, n); Hm[iu[0], iu[1]] = hu
    return Hm + Hm.T - torch.diag(Hm.diag()), int(md["rows"])
def ratio(e, s): return ((s * e).mean() / (s.mean() * e.mean())).item()
for l in map(int, sys.argv[1:]):
    P = f"model.language_model.layers.{l}.mlp."
    Wg, Wu, Wd = T(P + "gate_proj.weight"), T(P + "up_proj.weight"), T(P + "down_proj.weight")
    Hm, rows = H(f"model.language_model.layers.{l}.post_attention.input.safetensors")
    ev, V = torch.linalg.eigh(Hm); R = (V * ev.clamp_min(0).sqrt()).float()
    z = torch.randn(N, Wg.shape[1]); dd = (Wd ** 2).sum(0).double()
    out = {"layer": l, "rows": rows}
    for tag, sc in (("norm", 1 / math.sqrt(rows)), ("rawsum", 1.0)):
        x = (z @ R.T) * sc; g = x @ Wg.T; u = x @ Wu.T
        sg = torch.sigmoid(g); silu = g * sg; dsilu = sg * (1 + g * (1 - sg))
        for name, W, slope in (("up", Wu, silu), ("gate", Wg, u * dsilu)):
            s = (slope.double() ** 2).mean(0) * dd; e = (W.double() ** 2).mean(1)
            out[f"{name}_{tag}"] = round(ratio(e, s), 4)
    print(json.dumps(out), flush=True)
