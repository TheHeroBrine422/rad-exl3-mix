#!/usr/bin/env python3
"""check_exl3i_decode.py PLUGIN_EXL3IMPORT VEC.npz -- the plugin's exl3i decoder (common/exl3i_decode.c) against
(1) the reference oracle v3/infer/ref/exl3i_ref.c on random tiles at every rate 1/8 .. 8 in 1/8 steps, both codebooks
(values and states); (2) the GPU quantizer's vectors (dumpvec.py: packed words -> its indices / output values);
(3) at integer K with mul1, exllamav3's own oracle (common/exl3_decode.c exl3_trellis_decode, golden-checked);
(4) the rate spelling round trip. Exit 1 on any difference."""
import ctypes, os, subprocess, sys
import numpy as np
P, VEC = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__)); REF = os.path.join(HERE, "..", "ref", "exl3i_ref.c")
so_p, so_r = "/tmp/libexl3i_plug.so", "/tmp/libexl3i_ref.so"
subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", so_p, f"{P}/common/exl3i_decode.c",
                f"{P}/common/exl3_decode.c", f"-I{P}/common", "-lm"], check=True)
subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", so_r, REF, "-lm"], check=True)
A, R = ctypes.CDLL(so_p), ctypes.CDLL(so_r)
A.exl3i_decode_tile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
R.exl3i_decode_tile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
A.exl3i_rate_parse.argtypes = [ctypes.c_char_p]; A.exl3i_rate_str.restype = ctypes.c_char_p
A.exl3i_rate_str.argtypes = [ctypes.c_int, ctypes.c_char_p]; A.exl3i_mask.argtypes = [ctypes.c_int]
A.exl3_trellis_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
fails = 0; rng = np.random.default_rng(0)
# (1) vs the reference oracle
n1 = 0
for m in range(2, 129, 2):
    ka, mask = m // 16, A.exl3i_mask(m % 16)
    for cb in (0, 1):
        for _ in range(24):
            w = rng.integers(0, 2**32, m // 2, dtype=np.uint64).astype(np.uint32)
            va, sa, vr, sr = (np.zeros(256, np.uint16) for _ in range(4))
            assert A.exl3i_decode_tile(w.ctypes.data, m, cb, va.ctypes.data, sa.ctypes.data) == 0
            assert R.exl3i_decode_tile(w.ctypes.data, ka, mask, cb, sr.ctypes.data, vr.ctypes.data) == 0
            d = int((va != vr).sum() + (sa != sr).sum()); fails += d; n1 += 1
print(f"(1) vs reference oracle: {n1} tiles over 64 rates x 2 codebooks, {fails} differences")
# (2) GPU quantizer vectors
d = np.load(VEC); f2 = 0; n2 = 0
for key in sorted(k for k in d.files if k.endswith("_ka_mask")):
    tag = key[:-8]; ka, mk = map(int, d[key]); cb = 1 if tag.startswith("i8") else 0
    m = 16 * ka + bin(mk).count("1"); assert A.exl3i_mask(m % 16) == mk, (tag, m)
    idx, out, packed = d[tag + "_idx"], d[tag + "_out"], d[tag + "_packed"]
    for t in range(len(idx)):
        w = np.ascontiguousarray(packed[t]).view(np.uint32).copy(); va = np.zeros(256, np.uint16); sa = np.zeros(256, np.uint16)
        A.exl3i_decode_tile(w.ctypes.data, m, cb, va.ctypes.data, sa.ctypes.data)
        f2 += int((sa != idx[t]).sum()) + int((va.view(np.float16).astype(np.float32) != out[t]).sum()); n2 += 1
fails += f2; print(f"(2) GPU quantizer vectors: {n2} tiles, {f2} differences")
# (3) integer K, mul1 vs exllamav3's oracle (W_hat of a 2 x 3 tile tensor)
f3 = 0
for K in range(1, 9):
    kt, nt = 2, 3; tr = rng.integers(0, 2**16, kt * nt * 16 * K, dtype=np.uint64).astype(np.uint16)
    ref = np.zeros((16 * kt, 16 * nt), np.uint16); A.exl3_trellis_decode(tr.ctypes.data, kt, nt, K, 2, ref.ctypes.data)
    for tk in range(kt):
        for tn in range(nt):
            w = tr[(tk * nt + tn) * 16 * K:(tk * nt + tn + 1) * 16 * K].view(np.uint32).copy(); va = np.zeros(256, np.uint16)
            A.exl3i_decode_tile(w.ctypes.data, 16 * K, 0, va.ctypes.data, None)
            for p in range(256):
                L_, j = p // 8, p % 8; r = 2 * (L_ % 4) + [0, 1, 8, 9][j % 4]; c = L_ // 4 + 8 * (j // 4)
                f3 += int(va[p] != ref[16 * tk + r, 16 * tn + c])
fails += f3; print(f"(3) integer K1-8 mul1 vs exllamav3's oracle: {f3} differences")
# (4) rate spelling
f4 = 0
for m in range(2, 129, 2):
    b = ctypes.create_string_buffer(16); s = A.exl3i_rate_str(m, b)
    f4 += A.exl3i_rate_parse(s) != m
for bad in (b"2.250", b"02.5", b"2.3", b"0", b"8.125", b"-1", b"", b"2.5x"):
    f4 += A.exl3i_rate_parse(bad) != -1
fails += f4; print(f"(4) rate spelling: {f4} failures")
print("EXL3I DECODE", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
