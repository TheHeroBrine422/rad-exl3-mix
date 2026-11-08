#!/usr/bin/env python3
"""gate77.py OUT.json -- the i4 sub-1 gate of ../w77.md (run from the code copy on a box; builds its helpers here).
(a) decode: settle77's mul1x_i4 == radiance's oracle exl3i_ref.c + w53's exl3i_ref-i4.patch (cb 2) on all 65,536
    states; trellis.pattern(K) == exl3i_frac_mask for the gate rates; the salt == exl3i_salt at every position.
(b) search: tiles drawn like w53's qtf2i4check.py (N(0,1) : N(0,9) : round(2 N(0,1)) / 2 : zeros = 2048 : 512 : 256 : 8,
    x GATE_MULT, own seed); vit77.c with qtf2's tail-biting heuristic and fp16 costs vs w53's GPU rel_mse (+-1.5%);
    free-start fp32 (the sweep's search) reported too.
GATE_EXTRA=K[,K..] (lead 11-07: K0.125): only those K; (a) as above, (b) reported only (w53 has no GPU number there)."""
import ctypes, json, os, subprocess, sys
from fractions import Fraction
import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../.."))
sys.argv, OUT = sys.argv[:1], sys.argv[1]
import settle77, trellis                                   # noqa: E402 (settle77 installs mul1x_i4)

GPU = {0.25: 0.81941, 0.5: 0.61883, 0.75: 0.49917, 1.25: 0.35476, 1.5: 0.30732, 1.75: 0.27111, 2.5: 0.20242}
if os.environ.get("GATE_EXTRA"): GPU = {float(k): None for k in os.environ["GATE_EXTRA"].split(",")}
res = {"a": {}, "b": {}}

# (a) the oracle, patched in a private copy
ref = os.path.join(H, "ref"); os.makedirs(ref, exist_ok=True)
src = os.path.join(ref, "exl3i_ref.c")
if not os.path.exists(os.path.join(ref, "libexl3iref77.so")):
    subprocess.check_call(["cp", os.path.join(R, "v3/infer/ref/exl3i_ref.c"), src])
    pt = open(os.path.join(H, "exl3i_ref-i4.patch")).read()
    pt = pt[:pt.index("diff --git a/v3/infer/ref/test_ref.py")]           # the .c part only
    subprocess.run(["patch", src], input=pt.encode(), check=True)
    subprocess.check_call(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", src, "-o", os.path.join(ref, "libexl3iref77.so"), "-lm"])
L = ctypes.CDLL(os.path.join(ref, "libexl3iref77.so"))
L.exl3i_codebook.restype = ctypes.c_uint16; L.exl3i_salt.restype = ctypes.c_uint16; L.exl3i_frac_mask.restype = ctypes.c_int
orc = np.array([L.exl3i_codebook(k, 2) for k in range(65536)], np.uint16)
mine = settle77.codebook("mul1x_i4").astype(np.float16).view(np.uint16)
res["a"]["codebook_i4_differ"] = int((orc != mine).sum()); res["a"]["levels"] = int(len(np.unique(mine)))
o8 = np.array([L.exl3i_codebook(k, 1) for k in range(65536)], np.uint16)
res["a"]["codebook_i8_differ"] = int((o8 != settle77.codebook("mul1x_i8").astype(np.float16).view(np.uint16)).sum())
salt_c = [(0x9E37 * ((i & 15) + 1)) & 0xFFFF for i in range(256)]
res["a"]["salt_differ"] = sum(int(L.exl3i_salt(i)) != salt_c[i] for i in range(256))
pd = {}
for K in GPU:
    ka = int(K); n = round((K - ka) * 16); m = L.exl3i_frac_mask(n)
    pat = trellis.pattern(K); kk = [pat[i % len(pat)] for i in range(16)]
    pd[str(K)] = int(sum(kk[i] != ka + ((m >> i) & 1) for i in range(16)))
res["a"]["pattern_differ"] = pd
okA = res["a"]["codebook_i4_differ"] == 0 and res["a"]["levels"] == 16 and res["a"]["salt_differ"] == 0 and not any(pd.values())

# (b) the search, against w53's GPU numbers
lib = os.path.join(H, "libvit77.so")
if not os.path.exists(lib):
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", os.path.join(H, "vit77.c"), "-o", lib])
V = ctypes.CDLL(lib)
V.vit77_tiles.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
mult = int(os.environ.get("GATE_MULT", 2))
rng = np.random.default_rng(77)
N = 2048 * mult
X = np.concatenate([rng.standard_normal((N, 256)), rng.standard_normal((N // 4, 256)) * 3,
                    np.round(rng.standard_normal((N // 8, 256)) * 2) / 2, np.zeros((8 * mult, 256))]).astype(np.float32)
cb = np.ascontiguousarray(settle77.codebook("mul1x_i4"), dtype=np.float32)
okB = True
for K, g in GPU.items():
    pat = np.ascontiguousarray(trellis.pattern(K), dtype=np.int32); row = {"gpu": g}
    for tag, tb, f16 in (("tb_f16", 1, 1), ("free_f32", 0, 0)):
        V.vit77_set_f16(f16); out = np.empty_like(X)
        V.vit77_tiles(X.ctypes.data, out.ctypes.data, None, len(X), cb.ctypes.data, pat.ctypes.data, len(pat), tb)
        row[tag] = float(((out - X) ** 2).mean() / (X ** 2).mean())
    if K in (0.125, 0.5, 1.5):                                    # the sweep's own path (trellis.vit) == vit77 free fp32
        row["free_trellis_vit"] = float(((trellis.vit(X, cb, K) - X) ** 2).mean() / (X ** 2).mean())
    if g is not None:
        row["tb_over_gpu"] = row["tb_f16"] / g; row["free_over_gpu"] = row["free_f32"] / g
        okB &= abs(row["tb_over_gpu"] - 1) <= 0.015
    res["b"][str(K)] = row; print(json.dumps({"K": K, **row}), flush=True)
res["tiles"] = len(X); res["PASS_a"] = bool(okA); res["PASS_b"] = bool(okB) if any(v is not None for v in GPU.values()) else "n/a (no GPU number)"
res["verdict"] = "PASS" if okA and okB else "FAIL"
json.dump(res, open(OUT + ".part", "w"), indent=1); os.replace(OUT + ".part", OUT)
print(json.dumps({k: v for k, v in res.items() if k != "b"}))
