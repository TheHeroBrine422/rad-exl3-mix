#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""diag218.py KEY [CODES_DIR ...] [--dump=PREFIX] -- #218 diagnostic (CPU): the error of the released codes and of re-encoded codes against the
bf16 checkpoint's weight, in the stored codes' basis (W_hat_orig = Hb diag(1 / suh) W diag(1 / svh) Hb, W = [in, out]):
plain rms, and the relative output error tr(E^T M E) / tr(W^T M W) in the undamped metric M = Hb S H S Hb (the calib Hessian)."""
import json, os, sys

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
sys.path.insert(0, HERE)
from rxq import deq  # noqa: E402
from encode218 import hname  # noqa: E402

MODEL = "/data/workspace/models/qwen38-27b-bf16"; LV = "/data/w218/lv/2.50/trunk"; HESS = "/data/challenge/hessians/calib"
key, dirs = sys.argv[1], [d for d in sys.argv[2:] if not d.startswith("--dump=")]
dump = next((d[7:] for d in sys.argv[2:] if d.startswith("--dump=")), None)     # --dump=PREFIX: W_hat_orig f32 + suh f64 (reencode inputs)
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{MODEL}/{idx[key + '.weight']}", "pt") as f: W = f.get_tensor(key + ".weight").double().T.contiguous()   # [k, n]
lev = deq.Level(LV, "cpu"); raw = lev.raw(key); K = raw["trellis"].shape[2] / 16
table = torch.from_numpy(deq.lut(lev.codebook))
rel = deq.decode_hat(raw["trellis"], K, table).double()
suh, svh = raw["suh"].double(), raw["svh"].double()
k, n = W.shape
Hb = deq.hadamard128("cpu").double()
def had_k(x): return (Hb @ x.reshape(k // 128, 128, -1)).reshape(x.shape)          # left: blocks of 128 rows
def had_n(x): return (x.reshape(x.shape[0], -1, 128) @ Hb).reshape(x.shape)         # right
Wo = had_n(had_k(W / suh[:, None] / svh[None, :]))
with safe_open(f"{HESS}/{hname(key)}.safetensors", "pt") as f:
    up = f.get_tensor("hessian_upper").double().numpy(); md = f.metadata()
H = np.zeros((k, k)); iu = np.triu_indices(k); H[iu] = up; H = H + H.T - np.diag(np.diag(H)); H /= float(md["rows"])
H = torch.from_numpy(H)
M = had_n(had_k(H * suh[:, None] * suh[None, :]))
den = float((Wo * (M @ Wo)).sum())
if dump: Wo.float().numpy().tofile(dump + ".target"); suh.numpy().tofile(dump + ".suh")
def rep(name, q):
    E = q - Wo
    print(f"{name:10s} rms(E) {float(E.pow(2).mean().sqrt()):.4f}  rel out err {float((E * (M @ E)).sum()) / den:.5f}  "
          f"rms(q - rel) {float((q - rel).pow(2).mean().sqrt()):.4f}", flush=True)
print(f"{key} K {K} rms(W_hat_orig) {float(Wo.pow(2).mean().sqrt()):.4f}")
rep("released", rel)
for d in dirs:
    rep(os.path.basename(d.rstrip("/")), torch.from_numpy(np.fromfile(f"{d}/{key}.q", np.float32).reshape(k, n)).double())
d = 0.025 * float(torch.diagonal(H).mean())
print(f"damping as the tool had it (sigma mean diag H, before S): d S^2 mean {d * float((suh ** 2).mean()):.4e} vs mean diag(M) "
      f"{float(torch.diagonal(M).mean()):.4e}; suh^2 max / mean {float((suh ** 2).max() / (suh ** 2).mean()):.1f}; "
      f"diag(H) max / mean {float(torch.diagonal(H).max() / torch.diagonal(H).mean()):.1f}")
