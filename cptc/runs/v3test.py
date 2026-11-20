import ctypes, numpy as np, time, sys, os
L1 = ctypes.CDLL(sys.argv[1] + "/libviterbi.so"); L3 = ctypes.CDLL(sys.argv[1] + "/libviterbi3.so")
for L in (L1, L3):
    L.viterbi_tiles_tb_st.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
def run(L, x, cb, K, ntry):
    q = np.empty_like(x); st = np.zeros(x.shape, np.uint16)
    t = time.time(); L.viterbi_tiles_tb_st(x.ctypes.data, q.ctypes.data, st.ctypes.data, x.shape[0], cb.ctypes.data, K, ntry)
    return q, st, time.time() - t
rng = np.random.default_rng(3); ok = True
for K in (1, 2, 3, 4, 5, 6, 8):
    for name, cb in (("gauss", rng.standard_normal(65536)), ("coarse", np.round(rng.standard_normal(65536) * 3) / 3)):
        cb = cb.astype(np.float32); x = rng.standard_normal((96, 256)).astype(np.float32)
        for ntry in (1, 3):
            a = run(L1, x, cb, K, ntry); b = run(L3, x, cb, K, ntry)
            same = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]); ok &= same
            print(f"K{K} {name:6s} ntry{ntry}: {'IDENTICAL' if same else 'DIFF'}  legacy {a[2]:.2f}s  v3 {b[2]:.2f}s")
print("ALL IDENTICAL" if ok else "MISMATCH")
