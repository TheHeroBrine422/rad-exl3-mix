#!/usr/bin/env python3
"""test71.py [NTILES] -- self-checks of beamL.c / trellisL.py on synthetic tiles (no 27B data):
 1. full Viterbi at L = 16 == viterbi.c (trellis.vit: viterbi_tiles_pat free start; E3_TB path: viterbi_tiles_tb
    ntry 1), bit for bit, codebooks mul1_i8 and mul1, K = 2..6;
 2. codebook_L(cb, 16) == cb (the generalisation is today's table at L = 16);
 3. beam with M = 2^16 at L = 16 == full Viterbi (free and tail-biting), bit for bit;
 4. tail-biting holds: (s_0 >> K) == (s_255 & (2^(L-K) - 1)) on the decoded states, every L / M;
 5. reported: MSE on iid Gaussian tiles at the searched scale, per (L, M), and seconds per tile.
Needs W71_LIB (libbeamL.so) and E3_LIBVITERBI_DIR (a libviterbi.so built from viterbi.c) or the defaults."""
import ctypes, os, sys, time
import numpy as np

h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [h, os.path.join(h, ".."), os.path.join(h, "../..")]
import trellisL as T
import trellis, trellis2
if os.environ.get("E3_LIBVITERBI_DIR"):                      # a libviterbi.so built from viterbi.c on this machine
    trellis.LIB = ctypes.CDLL(os.path.join(os.environ["E3_LIBVITERBI_DIR"], "libviterbi.so"))
    trellis.LIB.viterbi_tiles_pat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)

rng = np.random.default_rng(0)
X = rng.standard_normal((N, 256)).astype(np.float32)
for cbn in ("mul1_i8", "mul1"):
    cb16 = np.ascontiguousarray(trellis2.codebook(cbn), dtype=np.float32)
    check(f"codebook_L({cbn}, 16) == table", np.array_equal(T.codebook_L(cbn, 16), cb16))
    for K in (2, 3, 4, 5, 6):
        sc = {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}[K]
        x = X * sc
        ref_free = trellis.vit(x, cb16, K)
        trellis.LIB.viterbi_tiles_tb.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        ref_tb = np.empty_like(x); xx = np.ascontiguousarray(x)
        trellis.LIB.viterbi_tiles_tb(xx.ctypes.data, ref_tb.ctypes.data, N, cb16.ctypes.data, K, 1)
        vf = T.search(x, cb16, 16, K, 0, 0); vt = T.search(x, cb16, 16, K, 0, 1)
        check(f"{cbn} K{K} Viterbi L16 free == viterbi_tiles_pat", np.array_equal(vf, ref_free), f"diff {int((vf != ref_free).sum())}")
        check(f"{cbn} K{K} Viterbi L16 tb == viterbi_tiles_tb(1)", np.array_equal(vt, ref_tb), f"diff {int((vt != ref_tb).sum())}")
        if cbn == "mul1_i8" and K in (2, 4, 6):
            bf = T.search(x[:6], cb16, 16, K, 65536, 0); bt = T.search(x[:6], cb16, 16, K, 65536, 1)
            check(f"{cbn} K{K} beam M=65536 L16 == Viterbi (free, tb)", np.array_equal(bf, vf[:6]) and np.array_equal(bt, vt[:6]),
                  f"diff {int((bf != vf[:6]).sum())} / {int((bt != vt[:6]).sum())}")

# 4 + 5: tail-biting and MSE per (L, M) on iid Gaussian tiles (scale: best of a small grid at L16 Viterbi)
cb16 = np.ascontiguousarray(trellis2.codebook("mul1_i8"), dtype=np.float32)
for K in (2, 3, 4):
    lib = T.lib(); n = min(N, 16); x = X[:n]
    best = None
    for s in np.arange(0.70, 1.31, 0.05):
        q = T.search(x * s, cb16, 16, K, 0, 1) / s; e = float(((q - x) ** 2).mean())
        if best is None or e < best[0]: best = (e, s)
    s = best[1]; row = []
    for L in (16, 18, 20):
        cb = T.codebook_L("mul1_i8", L, cb16)
        for M in ((0, 256, 1024, 4096) if L == 16 else (256, 1024, 4096) if L == 20 else (0, 256, 1024, 4096)):
            xx = np.ascontiguousarray(x * s, dtype=np.float32); out = np.empty_like(xx)
            idx = np.empty(xx.shape, dtype=np.uint32); t0 = time.time()
            lib.trellis_tiles(xx.ctypes.data, out.ctypes.data, idx.ctypes.data, None, n, cb.ctypes.data, L, K, M, 1)
            dt = (time.time() - t0) / n
            g = (1 << (L - K)) - 1
            tbok = bool(np.all((idx[:, 0] >> K) == (idx[:, 255] & g)))
            # every state is a shift of the previous one (the stream is consistent)
            sh = bool(np.all(((idx[:, :-1].astype(np.uint64) << K) & ((1 << L) - 1)) >> K == (idx[:, 1:].astype(np.uint64) >> K)))
            check(f"K{K} L{L} M{M} tail-biting + shift-consistent", tbok and sh)
            e = float(((out / s - x) ** 2).mean())
            row.append(f"L{L}/{'V' if M == 0 else M}: {e:.5f} ({dt*1e3:.0f} ms)")
    print(f"K{K} s={s:.2f} mse " + "  ".join(row), flush=True)
print("ALL PASS" if ok else "SOME FAIL")
