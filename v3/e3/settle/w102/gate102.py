#!/usr/bin/env python3
"""gate102.py OUT.json [--gauss NTILES] -- w102 #136-01 gate G1 (../w102.md): ring102.c against the existing searches.
  (a) ntry 0 (free start), every K of the test: values bit-identical to v3/e3/viterbi.c's viterbi_tiles_pat (the
      old proxy's search, sub-1 salted 0-bit reads included);
  (b) n = 1, ntry 1 and 16, integer K 1 / 2: bit-identical to viterbi_tiles_tb; n = 4, K2: bit-identical to
      chain/viterbi.c's viterbi_chain_tb (#102);
  (c) every tail-biting output (sub-1 included) is a valid ring: ring_check() finds 0 shift / wrap violations, and the
      values decode from the returned states (cb[s ^ salt]).
  Timing per arm (seconds per 1,000 tiles, one thread) is printed. With --gauss N: also the i.i.d. Gaussian table
  (mse ratios of tb16 / chain to tb1 per K) -- run only after the rule commit (it is data)."""
import ctypes, json, os, sys, time
from fractions import Fraction
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.join(H, "..", "..")
sys.path[:0] = [E3]
import trellis  # noqa: E402

R = ctypes.CDLL(os.path.join(H, "libring102.so"))
V = ctypes.CDLL(os.path.join(E3, "libviterbi.so"))
C = ctypes.CDLL(os.path.join(E3, "chain", "libviterbi.so"))
vp, ci = ctypes.c_void_p, ctypes.c_int
R.ring_tb.argtypes = [vp, vp, vp, ci, ci, vp, vp, ci, ci]
R.ring_check.argtypes = [vp, ci, ci, vp, ci]; R.ring_check.restype = ci
V.viterbi_tiles_pat.argtypes = [vp, vp, ci, vp, vp, ci]
V.viterbi_tiles_tb.argtypes = [vp, vp, ci, vp, ci, ci]
C.viterbi_chain_tb.argtypes = [vp, vp, vp, ci, ci, vp, vp, ci, ci]


def pat(K):
    return np.ascontiguousarray(trellis.pattern(K), dtype=np.int32)


def ring(x, cb, K, n, ntry):
    """x [ntiles, 256] -> (values [ntiles, 256], states) with chains of n tiles; ntry 0 = free start"""
    nt = x.shape[0]; assert nt % n == 0
    xr = np.ascontiguousarray(x, dtype=np.float32).reshape(nt // n, 256 * n)
    out = np.empty_like(xr); idx = np.empty(xr.shape, dtype=np.uint16); p = pat(K)
    R.ring_tb(xr.ctypes.data, out.ctypes.data, idx.ctypes.data, nt // n, 256 * n, cb.ctypes.data, p.ctypes.data, len(p), ntry)
    return out.reshape(nt, 256), idx


def check(idx, K, n):
    p = pat(K); L = 256 * n
    return int(R.ring_check(idx.ctypes.data, idx.size // L, L, p.ctypes.data, len(p)))


def main():
    out = sys.argv[1]
    gauss = int(sys.argv[sys.argv.index("--gauss") + 1]) if "--gauss" in sys.argv else 0
    sys.path.insert(0, os.path.join(H, "..", "w64")); import settle64  # noqa: E402  (the fork's mul1 values)
    cb = np.ascontiguousarray(settle64.codebook("mul1x_i8"), dtype=np.float32)
    x = np.random.default_rng(102).standard_normal((32, 256)).astype(np.float32)
    res = {"a": {}, "b": {}, "c": {}, "sec_per_1k_tiles": {}, "PASS": True}
    for K in (0.125, 0.25, 0.5, 1, 1.5, 2):
        q, _ = ring(x, cb, K, 1, 0)
        ref = np.empty_like(x); p = pat(K)
        V.viterbi_tiles_pat(x.ctypes.data, ref.ctypes.data, len(x), cb.ctypes.data, p.ctypes.data, len(p))
        res["a"][str(K)] = ok = bool(np.array_equal(q, ref)); res["PASS"] &= ok
    for K in (1, 2):
        for nt in (1, 16):
            q, _ = ring(x, cb, K, 1, nt); ref = np.empty_like(x)
            V.viterbi_tiles_tb(x.ctypes.data, ref.ctypes.data, len(x), cb.ctypes.data, K, nt)
            res["b"][f"tb K{K} ntry{nt}"] = ok = bool(np.array_equal(q, ref)); res["PASS"] &= ok
    q, _ = ring(x, cb, 2, 4, 1); ref = np.empty((8, 1024), dtype=np.float32); p = pat(2)
    C.viterbi_chain_tb(np.ascontiguousarray(x.reshape(8, 1024)).ctypes.data, ref.ctypes.data, None, 8, 1024, cb.ctypes.data, p.ctypes.data, 1, 1)
    res["b"]["chain n4 K2"] = ok = bool(np.array_equal(q, ref.reshape(32, 256))); res["PASS"] &= ok
    for K in (0.125, 0.5, 1):
        for n, nt in ((1, 1), (1, 16), (4, 1), (4, 16)):
            t0 = time.time(); q, idx = ring(x, cb, K, n, nt); dt = time.time() - t0
            p = pat(K); dec = cb[idx.reshape(-1, 256 * n) ^ np.array([0 if p[i % len(p)] else (0x9E37 * ((i & 15) + 1)) & 0xFFFF
                                                                      for i in range(256 * n)], dtype=np.uint16)].reshape(q.shape)
            bad = check(idx, K, n)
            res["c"][f"K{K} n{n} ntry{nt}"] = {"violations": bad, "decodes": bool(np.array_equal(dec, q))}
            res["PASS"] &= bad == 0 and bool(np.array_equal(dec, q))
            res["sec_per_1k_tiles"][f"K{K} n{n} ntry{nt}"] = round(dt / len(x) * 1000, 1)
        t0 = time.time(); ring(x, cb, K, 1, 0); res["sec_per_1k_tiles"][f"K{K} free"] = round((time.time() - t0) / len(x) * 1000, 1)
    if gauss:
        g = np.random.default_rng(7).standard_normal((gauss, 256)).astype(np.float32)
        tab = {}
        for K in (0.125, 0.5, 1):
            m = {}
            for name, n, nt in (("free", 1, 0), ("tb1", 1, 1), ("tb16", 1, 16), ("chain2", 2, 1), ("chain4", 4, 1), ("chain8", 8, 1), ("chain4x16", 4, 16)):
                q, _ = ring(g, cb, K, n, nt); m[name] = float(((q - g) ** 2).mean())
            tab[str(K)] = {"mse": m, "ratio_to_tb1": {k: v / m["tb1"] for k, v in m.items()}}
        res["gauss"] = {"tiles": gauss, "codebook": "mul1x_i8 (unscaled)", "table": tab}
    print(json.dumps(res, indent=1))
    with open(out + ".part", "w") as f: json.dump(res, f, indent=1)
    os.replace(out + ".part", out)


if __name__ == "__main__":
    main()
