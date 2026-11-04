#!/usr/bin/env python3
"""packcheck.py QDIR KEY [WQFILE] -- w11: is the dequantized weight the eval uses (QDIR/wq/KEY or WQFILE, (out, in) f16)
what the pack (QDIR's q-LNN suh / svh, f16) can represent?  W_q (in, out) = diag(suh) Hl Qc Hr diag(svh), Hl / Hr
blockwise 128 Sylvester Hadamards (quantize.py preapply_had_l/r). Un-rotates W_q with the PACK's suh / svh and reports
how close Qc is to the i8 codebook grid (multiples of 4 k_inv, k_inv = f16 0x1eee, |m| <= 127): residual in grid steps
(a pack-decodable W_q gives ~0 up to the f16 rounding of the saved weight). CPU, numpy only."""
import sys, json, collections
import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open
q, key = sys.argv[1:3]; wqf = sys.argv[3] if len(sys.argv) > 3 else f"{q}/wq/{key}.safetensors"
l = int(key.split(".")[3])
with safe_open(f"{q}/q-L{l:02d}.safetensors", "np") as f:
    su = f.get_tensor(key + ".suh").astype(np.float64); sv = f.get_tensor(key + ".svh").astype(np.float64)
W = load_file(wqf)["weight"].astype(np.float64).T            # (in, out)
def had(n):
    h = np.array([[1.0]])
    while h.shape[0] < n: h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)
H = had(128); k, n = W.shape
X = W / su[:, None] / sv[None, :]
X = (H @ X.reshape(k // 128, 128, n)).reshape(k, n)          # Hl^-1 = Hl (symmetric orthonormal)
X = (X.reshape(k, n // 128, 128) @ H).reshape(k, n)
d = 4 * float(np.frombuffer(np.uint16(0x1eee).tobytes(), np.float16)[0])   # i8 codebook step: 4 k_inv (exl3cb)
g = X / d; gres = g - np.round(g)
grid = {"i8_step": d, "grid_resid_rms_steps": float(np.sqrt((gres ** 2).mean())), "grid_resid_max_steps": float(np.abs(gres).max()),
        "share_within_0.05_step": float((np.abs(gres) < 0.05).mean()), "max_code": int(np.abs(np.round(g)).max())}
r = np.round(X, 3).ravel(); top = np.array([v for v, _ in collections.Counter(r).most_common(256)]); top.sort()
idx = np.clip(np.searchsorted(top, X.ravel()), 1, len(top) - 1)
near = np.where(np.abs(X.ravel() - top[idx - 1]) < np.abs(X.ravel() - top[idx]), top[idx - 1], top[idx])
res = X.ravel() - near
print(json.dumps({"key": key, "q": q, "wq": wqf, "shape": [k, n], "distinct_round3": int(len(set(r.tolist()))),
                  "share_within_1e-2_of_top256": float((np.abs(res) < 1e-2).mean()),
                  "rel_rms_resid": float(np.sqrt((res ** 2).mean() / (X ** 2).mean())), "rms_codes": float(np.sqrt((X ** 2).mean())),
                  "suh_absmax": float(np.abs(su).max()), "svh_absmax": float(np.abs(sv).max()), **grid}))
