#!/usr/bin/env python3
"""cbsmoke.py VARIANT [NTILES] -- check of the rounded-codebook search (exl3cb, VARIANT i8 or i4) on the GPU.
Seeded Gaussian 256-value tiles quantized with plain mul1 (prebuilt exllamav3) and with the rounded codebook, per K:
  grid   every returned value lies on the variant's grid (i8: integer multiples of 4 k_inv, |q| <= 127; i4: the 16
         half-integer levels of 2.6/7.5) -- i.e. the values LDLQ feeds back are what an int kernel multiplies
  ratio  rounded-codebook mse / plain mse (E3's CPU numbers: i8 1.000-1.004 to K6, 1.25 K7, 4.5 K8; i4 1.01 K2, 1.07 K3)
Exit 1 if any value is off the grid or a ratio exceeds 1.05 where E3 expects it to be free (i8 K <= 6, i4 K <= 2).
Env EXL3CB_SRC (the fork's exllamav3_ext), PYTORCH_ROCM_ARCH, ROCM_HOME, TORCH_EXTENSIONS_DIR."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3cb"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3frac"))
V = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
import exllamav3.modules.quant.exl3_lib.quantize as Q
x = torch.from_numpy(np.random.default_rng(11).standard_normal((N, 256)).astype(np.float32)).cuda()
KS = [1, 2, 2.5, 3, 4, 5, 6, 7, 8] if V == "i8" else [1, 1.5, 2, 2.25, 2.5, 3, 4]
import exl3frac; Q.ext = exl3frac._Ext(Q.ext, exl3frac.load(os.environ["EXL3CB_SRC"])); Q.frac_k = exl3frac.frac_k
plain = {K: float(((Q.quantize_tiles(x, {"K": K, "mul1": True})[0] - x) ** 2).mean()) for K in KS}
import exl3cb; exl3cb.install(os.environ["EXL3CB_SRC"], V)
s4 = 4.0 * float(np.float16(np.frombuffer(np.uint16(0x1eee).tobytes(), np.float16)[0]))
bad = []
for K in KS:
    q = Q.quantize_tiles(x, {"K": K, "mul1": True})[0]
    r = float(((q - x) ** 2).mean()) / plain[K]
    v = q.double().cpu().numpy().ravel()
    if V == "i8":
        g = v / s4; off = np.abs(g - np.round(g)).max() * s4; lev = np.unique(np.round(g)); ok = off < 2e-3 and np.abs(lev).max() <= 127
    else:
        st = 2.6 / 7.5; g = v / st - 0.5; off = np.abs(g - np.round(g)).max() * st; lev = np.unique(np.round(g)); ok = off < 2e-3 and len(lev) <= 16
    print(f"K{K:<4} ratio {r:.3f}  levels used {len(lev):4d}  max off-grid {off:.2e}  {'ok' if ok else 'OFF GRID'}", flush=True)
    if not ok: bad.append(f"K{K} off grid")
    if (V == "i8" and K <= 6 or V == "i4" and K <= 2) and r > 1.05: bad.append(f"K{K} ratio {r:.3f}")
print(f"CBSMOKE {V} " + ("OK" if not bad else "FAIL: " + ", ".join(bad)))
sys.exit(1 if bad else 0)
