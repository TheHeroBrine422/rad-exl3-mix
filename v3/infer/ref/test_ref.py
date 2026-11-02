#!/usr/bin/env python3
"""test_ref.py [VEC.npz] -- the exl3i oracle (exl3i_ref.c) against (1) an independent numpy mul1 / mul1i8 codebook over
all 65,536 states, (2) the GPU quantizer's vectors (dumpvec.py): for every rate, decoding the packed words must give
exactly the quantizer's indices and output values. Builds libexl3iref.so next to this file."""
import ctypes, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
so = os.path.join(HERE, "libexl3iref.so")
subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "exl3i_ref.c"), "-lm"], check=True)
L = ctypes.CDLL(so)
L.exl3i_codebook.restype = ctypes.c_uint16; L.exl3i_codebook.argtypes = [ctypes.c_uint32, ctypes.c_int]
L.exl3i_decode_tile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
# (1) codebook vs numpy (same transcription as v3/nf/check/i8round.py)
s = np.arange(65536, dtype=np.uint64); x = (s * 0x83DCD12D) & 0xFFFFFFFF
bs = sum(((x >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
f16 = lambda u: np.array([u], np.uint16).view(np.float16)[0]
KINV, KBIAS = f16(0x1EEE), f16(0xC931)
v = ((1024 + bs) * float(KINV) + float(KBIAS)).astype(np.float16)
s4 = np.float32(4.0) * np.float32(KINV)
q = np.clip(np.rint(v.astype(np.float32) * (np.float32(1.0) / s4)), -127, 127).astype(np.float32)
vi8 = (q * s4).astype(np.float16)
c0 = np.array([L.exl3i_codebook(int(k), 0) for k in range(65536)], np.uint16)
c1 = np.array([L.exl3i_codebook(int(k), 1) for k in range(65536)], np.uint16)
print(f"codebook mul1: {int((c0 != v.view(np.uint16)).sum())} of 65536 differ; mul1i8: {int((c1 != vi8.view(np.uint16)).sum())} differ")
fails = int((c0 != v.view(np.uint16)).sum() + (c1 != vi8.view(np.uint16)).sum())
# (2) GPU vectors
if len(sys.argv) > 1:
    d = np.load(sys.argv[1])
    for key in sorted(k for k in d.files if k.endswith("_ka_mask")):
        tag = key[:-8]; ka, m = map(int, d[key]); i8 = tag.startswith("i8")
        idx, out, packed = d[tag + "_idx"], d[tag + "_out"], d[tag + "_packed"]
        bad_s = bad_v = 0
        for t in range(len(idx)):
            w = np.ascontiguousarray(packed[t]).view(np.uint32).copy()
            st = np.zeros(256, np.uint16); va = np.zeros(256, np.uint16)
            assert L.exl3i_decode_tile(w.ctypes.data, ka, m, int(i8), st.ctypes.data, va.ctypes.data) == 0
            bad_s += int((st != idx[t]).sum())
            bad_v += int((va.view(np.float16).astype(np.float32) != out[t]).sum())
        fails += bad_s + bad_v
        print(f"{tag:<12} ka {ka} mask {m:04x}: states {bad_s} differ, values {bad_v} differ")
print("ORACLE", "PASS" if fails == 0 else f"FAIL ({fails})")
