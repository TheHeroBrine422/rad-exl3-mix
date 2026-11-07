#!/usr/bin/env python3
"""check_i4lut.py OUTDIR [RXM] -- rxq.deq's i4 codebook (the int4 grid every i4 level decodes with) against w53's C oracle:
v3/infer/ref/exl3i_ref.c with local/workers/instr3/w53-results/exl3i_ref-i4.patch applied in OUTDIR (the repo copy is
not touched), built with gcc -O2 -ffp-contract=off; exl3i_codebook(state, cb) for cb 0 / 1 / 2 (mul1 / i8 / i4) over
all 65,536 states, fp16 bits equal. Last line "I4LUT CHECK PASS|FAIL"."""
import ctypes, os, shutil, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
O = sys.argv[1]; RXM = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import deq  # noqa: E402

os.makedirs(O, exist_ok=True)
src = os.path.join(O, "exl3i_ref.c")
shutil.copy(f"{RXM}/v3/infer/ref/exl3i_ref.c", src)
patch = f"{RXM}/local/workers/instr3/w53-results/exl3i_ref-i4.patch"
# the patch's exl3i_ref.c hunks only (it also touches test_ref.py): git apply in OUTDIR on a one-file tree
subprocess.run(["git", "init", "-q", O], check=True)
os.makedirs(f"{O}/v3/infer/ref", exist_ok=True); shutil.move(src, f"{O}/v3/infer/ref/exl3i_ref.c")
subprocess.run(["git", "-C", O, "apply", "--include=v3/infer/ref/exl3i_ref.c", patch], check=True)
so = f"{O}/libref.so"
subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", so, f"{O}/v3/infer/ref/exl3i_ref.c", "-lm"], check=True)
L = ctypes.CDLL(so); L.exl3i_codebook.restype = ctypes.c_uint16; L.exl3i_codebook.argtypes = [ctypes.c_uint32, ctypes.c_int]
bad = {}
for cb, name in ((0, "mul1"), (1, "i8"), (2, "i4")):
    ref = np.array([L.exl3i_codebook(s, cb) for s in range(65536)], np.uint16)
    mine = deq.lut(name).astype(np.float16).view(np.uint16)
    bad[name] = int((ref != mine).sum())
    print(f"{name}: {bad[name]} of 65536 states differ; distinct values {len(np.unique(mine))}")
print(f"I4LUT CHECK {'PASS' if not any(bad.values()) else 'FAIL'} {bad}")
sys.exit(0 if not any(bad.values()) else 1)
