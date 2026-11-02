#!/usr/bin/env python3
"""prebuild.py SRC -- compile ourq's side extensions on a CUDA box (exl3frac, exl3cb i8, exl3frac2 i8), in the order
ourq.py installs them, into $TORCH_EXTENSIONS_DIR, so the first quantization does not pay for it."""
import os, sys, time
import torch
E = "/workspace/rad-exl3-mix/v3/e3"
for d in ("exl3frac", "exl3cb", "exl3frac2"): sys.path.insert(0, f"{E}/{d}")
src = sys.argv[1]; t0 = time.time()
import exl3frac; exl3frac.install(src); print(f"exl3frac {time.time() - t0:.0f}s", flush=True)
import exl3cb; exl3cb.install(src, "i8"); print(f"exl3cb i8 {time.time() - t0:.0f}s", flush=True)
import exl3frac2; exl3frac2.install(src, "i8"); print(f"exl3frac2 i8 {time.time() - t0:.0f}s", flush=True)
print("PREBUILD DONE")
