#!/usr/bin/env python3
"""smoke.py OUT.jsonl [NTILES] -- w45 / #94 rerun, step 1: does the CPU trellis + codebook core under settle.py
reproduce the GPU smoke (cbsmoke.py, LOG 2026-11-01 ~23:20: int8-rounded mul1 / mul1 = x1.024 K7, x1.347 K8)?

The smoke's tensors: cbsmoke.py's seeded Gaussian tiles, np.random.default_rng(11).standard_normal((4096, 256)) as
float32, quantized directly (no LDLQ, no transforms, no scale: tiles of sd 1 against a codebook of sd ~1) by
exllamav3's quantize_tiles, mul1 vs the EXL3_MUL1_I8 build. Here the same tiles through trellis.vit with
  settle    settle.py's codebooks (SETTLE_CB): trellis.py's standardised mul1 and mul1_i8 (step 3.5/127 sd)
  exl3      exllamav3's exact values (cbx.py): mul1 = fp16 fma(1024 + bytesum, k_inv, k_bias), i8 = rint(v / 4 k_inv)
            * 4 k_inv (one int8 step = 4 bytesum units, the fork's mul1_i8_round)
and two searches: free start state (settle.py's, trellis.vit default) and tail-biting with exllamav3's heuristic
(E3_TB=1, integer K only: what the GPU runs), plus "free16": the free search with fp16 costs, inputs and codebook
values (viterbi_set_f16, #57's emulation of the GPU search's arithmetic). Also at tile scale c != 1 (the tiles x c, error / c^2), since the real
quantizer feeds the trellis tiles of RMS s_exl x m (0.8-0.9 at high K), not 1.
Env SMOKE_KS (default 6,7,8), SMOKE_SCALES (default 1), SMOKE_SEARCH (default free,tb1,free16; tb1 only at integer K)."""
import json, os, sys, time
import numpy as np

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.environ.get("E3_DIR", ".")]
import trellis, trellis2, cbx

out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
x = np.random.default_rng(11).standard_normal((4096, 256)).astype(np.float32)[:n]     # cbsmoke.py's tiles
KS = [float(k) for k in os.environ.get("SMOKE_KS", "6,7,8").split(",")]
SC = [float(c) for c in os.environ.get("SMOKE_SCALES", "1").split(",")]
pairs = {"settle": ("mul1", "mul1_i8"), "exl3": ("mul1x", "mul1x_i8")}
for fam, (a, b) in pairs.items():
    cba, cbb = (np.ascontiguousarray(cbx.codebook(c), dtype=np.float32) for c in (a, b))
    for K in KS:
        for search in os.environ.get("SMOKE_SEARCH", "free,tb1,free16").split(","):
            if search == "tb1" and not K.is_integer(): continue
            os.environ["E3_TB"] = "1" if search == "tb1" else "0"
            trellis.LIB.viterbi_set_f16(1 if search == "free16" else 0)
            for c in SC:
                t0 = time.time()
                ma = float(((trellis.vit(x * c, cba, K) / c - x) ** 2).mean())
                mb = float(((trellis.vit(x * c, cbb, K) / c - x) ** 2).mean())
                o = {"set": fam, "cb": [a, b], "K": K, "search": search, "scale": c, "tiles": n, "mse_mul1": ma,
                     "mse_i8": mb, "ratio": mb / ma, "sec": round(time.time() - t0, 1)}
                print(json.dumps(o), flush=True)
                with open(out, "a") as f: f.write(json.dumps(o) + "\n")
