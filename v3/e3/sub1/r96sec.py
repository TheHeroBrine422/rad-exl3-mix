#!/usr/bin/env python3
"""r96sec.py LAYER TENSOR -- #96 side check: how much of a stacked tensor's S = 128 row-slab ceiling (rowslab2.py's
"out" weights) a K per section already gets (ssm_qkv: q 2048 / k 2048 / v 6144 rows; attn_qg: q vs gate, interleaved
per 256-row head; ffn_gate_up: gate vs up). Sections differ in size, so densities d_i = w_i / n_i:
gain = 0.5 log2(AM_n(d) / GM_n(d)) with row-count weights. Prints one JSON line (sec = per section, S128 = per slab)."""
import json, math, os, sys
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import e3
from rowslab2 import load_h32, HOUT
torch.set_grad_enabled(False)
layer, tensor = int(sys.argv[1]), sys.argv[2]
names, hn = e3.ROUTE[tensor]
W = e3.load_w(layer, names).float(); H = load_h32(layer, hn)
ref = ((W @ H) * W).sum(1).double().numpy(); N = W.shape[0]; del W, H
h = np.concatenate([safe_open(f"{HOUT}/{e3.P}{layer}.{n}.safetensors", "pt").get_tensor("hout_diag").double().numpy() for n in names])
w = np.clip(h, 0, None) * ref
lab = np.zeros(N, int)
if tensor == "ssm_qkv": lab[2048:4096] = 1; lab[4096:] = 2
elif tensor == "attn_qg": lab = (np.arange(N) // 256) % 2
elif tensor == "ffn_gate_up": lab[N // 2:] = 1


def gain_dens(wsum, n):
    d = wsum / n; p = n / n.sum()
    return 0.5 * math.log2((p * d).sum() / math.exp((p * np.log(np.clip(d, 1e-300, None))).sum()))


ids = np.unique(lab)
sec = gain_dens(np.array([w[lab == i].sum() for i in ids]), np.array([(lab == i).sum() for i in ids], float))
s128 = gain_dens(w.reshape(-1, 128).sum(1), np.full(N // 128, 128.0))
share = [float(w[lab == i].sum() / w.sum()) for i in ids]
print(json.dumps({"layer": layer, "w": tensor, "sec": sec, "S128": s128, "mass_share": share}), flush=True)
