#!/usr/bin/env python3
"""i8round.py -- the int8-rounded mul1 codebook: quantizer (exl3cb's mul1_i8_round, codebook.cuh) vs the int8 kernel
(nfq_wdec_i8), over all 65,536 trellis states. Both decode v = fp16_fma(fp16(1024 + bytesum(s * 0x83DCD12D)), K_INV,
K_BIAS) (bit-exact, B19), then q = clamp(rint(v * (1 / (4 K_INV))), -127, 127) in fp32. The quantizer's search sees
fp16(q * 4 K_INV); the kernel multiplies q and applies 4 K_INV in fp32 in the epilogue."""
import numpy as np
f16 = lambda u: np.array([u], np.uint16).view(np.float16)[0]
KINV, KBIAS = f16(0x1EEE), f16(0xC931)
s = np.arange(65536, dtype=np.uint64)
x = (s * 0x83DCD12D) & 0xFFFFFFFF
bs = sum(((x >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
h = (1024 + bs).astype(np.float16)                                   # exact (integers < 2048)
v = (h.astype(np.float64) * float(KINV) + float(KBIAS)).astype(np.float16)   # one rounding = fp16 fma
s4 = np.float32(4.0) * np.float32(KINV)
inv_q = np.float32(1.0) / s4                                          # quantizer: 1.0f / s4
inv_k = np.float32(1.0) / (np.float32(4.0) * np.float32(KINV))        # kernel: 1.f / nfq_wscale()
rq = lambda inv: np.clip(np.rint(v.astype(np.float32) * inv), -127, 127).astype(np.int32)
q1, q2 = rq(inv_q), rq(inv_k)
print("states", len(s), "distinct v", len(np.unique(v)), "q mismatches", int((q1 != q2).sum()),
      "q range", int(q1.min()), int(q1.max()), "clamped", int((np.abs(np.rint(v.astype(np.float32) * inv_q)) > 127).sum()))
seen = (q1.astype(np.float32) * s4).astype(np.float16).astype(np.float64)      # what the search optimised against
used = q1.astype(np.float64) * float(s4)                                       # what the kernel multiplies (fp32 path)
rel = np.abs(seen - used) / np.maximum(np.abs(used), 1e-12)
print(f"search value vs kernel value: max rel diff {rel[used != 0].max():.2e} (fp16 rounding of q x 4 K_INV), "
      f"rms over states {np.sqrt(np.mean((seen - used) ** 2)) / np.sqrt(np.mean(used ** 2)):.2e}")
