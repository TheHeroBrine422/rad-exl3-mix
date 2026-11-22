#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qerr218.py OUT.json [--layers 8,23,40,55] -- #218: each probe tensor's quantization step in the stored codes' basis,
delta = rms(W_hat_released - W_hat_orig) (W_hat_orig = Hb diag(1 / suh) W diag(1 / svh) Hb from the bf16 checkpoint, W = [in, out]),
the unit of encode218.py --step quant (eta = the step's rms as a fraction of delta). CPU."""
import json, os, sys

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
from rxq import deq  # noqa: E402

MODEL = "/data/workspace/models/qwen38-27b-bf16"; LV = "/data/w218/lv/2.50/trunk"
out = sys.argv[1]
layers = {int(x) for x in (sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--layers" else "8,23,40,55").split(",")}
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
lev = deq.Level(LV, "cpu"); table = torch.from_numpy(deq.lut(lev.codebook)); Hb = deq.hadamard128("cpu")
res = {}
for key in lev.keys():
    if int(key.split(".")[3]) not in layers: continue
    with safe_open(f"{MODEL}/{idx[key + '.weight']}", "pt") as f: W = f.get_tensor(key + ".weight").float().T.contiguous()
    raw = lev.raw(key); K = raw["trellis"].shape[2] / 16
    q = deq.decode_hat(raw["trellis"], K, table)
    k, n = W.shape
    Wo = W / raw["suh"].float()[:, None] / raw["svh"].float()[None, :]
    Wo = (Hb @ Wo.reshape(k // 128, 128, n)).reshape(k, n)
    Wo = (Wo.reshape(k, n // 128, 128) @ Hb).reshape(k, n)
    res[key] = {"K": K, "delta": float((q - Wo).pow(2).mean().sqrt()), "rms_w": float(Wo.pow(2).mean().sqrt())}
    print(key, res[key], flush=True)
json.dump(res, open(out, "w"), indent=1)
