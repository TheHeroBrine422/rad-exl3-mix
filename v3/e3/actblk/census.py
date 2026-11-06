#!/usr/bin/env python3
"""actblk/census.py HESSDIR [LAYERS] [TENSOR_H] -- #103 follow-up: the spread of 128-block traces of the calib H_in
for one input of every layer (default mlp.down_proj.input), read from the packed upper triangle by slicing only its
diagonal (no full H). One JSON line per layer: block trace max / min, p90 / p10, coefficient of variation.
The block trace is invariant under EXL3's in-block rotation, so this is the mass the blk128 order sorts."""
import json, sys
import numpy as np
from safetensors import safe_open

P = "model.language_model.layers."
hd = sys.argv[1]
layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else range(64)
hn = sys.argv[3] if len(sys.argv) > 3 else "mlp.down_proj.input"
import os
for L in layers:
    if not os.path.exists(f"{hd}/{P}{L}.{hn}.safetensors"): continue
    with safe_open(f"{hd}/{P}{L}.{hn}.safetensors", "np") as f:
        k = int(f.metadata()["k"]); s = f.get_slice("hessian_upper")
        d = np.array([s[i * k - i * (i - 1) // 2:i * k - i * (i - 1) // 2 + 1][0] for i in range(k)], dtype=np.float64)
    m = d.reshape(k // 128, 128).sum(1)
    print(json.dumps({"layer": L, "h": hn, "k": k, "blocks": len(m), "max_min": float(m.max() / max(m.min(), 1e-300)),
                      "p90_p10": float(np.quantile(m, 0.9) / max(np.quantile(m, 0.1), 1e-300)),
                      "cv": float(m.std() / m.mean())}), flush=True)
