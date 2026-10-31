#!/usr/bin/env python3
"""rowslab.py LAYER TENSOR [OUT.jsonl] -- E3c: is K per output-row slab worth it? (CPU, no quantization needed)

At high rate the quantization error of row r is err_r ~ c * ref_r * 2^(-2 b_r), ref_r = W_r H W_r^T (the input-side
Hessian norm of the row; the trellis quantizes scale-normalised rows, so the constant is shared). Spending bits per
slab of S rows optimally (reverse water-filling) instead of uniformly saves
    gain_bits = 0.5 * log2( AM(ref_slab) / GM(ref_slab) )        ref_slab = sum of ref_r over the slab
bits per weight at the same total error -- the ceiling for row-slab K with continuous bits (integer K steps get
somewhat less). Input-side only: output-side importance (YAQA's H_out) would add to the spread.
Reported for S = 1 (per row), 16, 128, 256, 1024. A stacked gate_up is gate rows then up rows.
"""
import json, math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3
torch.set_grad_enabled(False)

layer, tensor = int(sys.argv[1]), sys.argv[2]
names, hn = e3.ROUTE[tensor]
e3.ROWS = 0
W = e3.load_w(layer, names).float(); H = e3.load_h(layer, hn).float()
ref = ((W @ H) * W).sum(1).double().clamp(min=1e-30).numpy()
o = {"layer": layer, "w": tensor, "N": W.shape[0], "K": W.shape[1]}
for S in (1, 16, 128, 256, 1024):
    if W.shape[0] % S: continue
    r = ref.reshape(-1, S).sum(1)
    o[f"gain_S{S}"] = 0.5 * math.log2(r.mean() / math.exp(np.log(r).mean()))
print(json.dumps(o), flush=True)
if len(sys.argv) > 3:
    with open(sys.argv[3], "a") as f: f.write(json.dumps(o) + "\n")
