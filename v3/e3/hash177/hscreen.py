#!/usr/bin/env python3
"""hscreen.py -- #177 codebook screen on iid N(0,1) 16x16 tiles (no weights, no calib): plain bitshift-trellis MSE
(vitL.c, free initial state), the codebook's scale fitted (a 5-point grid around its unit-sd value + a parabola through
the best three). Codebook specs (each a table over the 2^L states, standardised):
  m1i8                 mul1i8: bytesum(s * 0x83DCD12D mod 2^32) rounded to the int8 grid (+-127 = 3.5 sd) -- the V3 codebook
  m1                   mul1 (V2's fp16 codebook, unrounded)
  h:A:B:MASK           IQR-style hash: S = bytesum(((s * A + B) mod 2^32) & MASK); A of any width (16-bit A = the
                       v_mad_u32_u16 form, 24-bit = v_mad_u32_u24); MASK 0x3F3F3F3F = IQR's, 0xFCFCFCFC = top 6 bits a byte
  hg:A:B:MASK          the same S remapped to Gaussian quantiles of its own CDF, int8 grid (a 256-entry remap table)
  hc:C:A:B:MASK        cubic shaping v + C v^3 of the standardised S, int8 grid (2 packed FMAs a pair)
  hyb:A:B:IB[:FILE]    hash -> LUT: value = LUT[top IB bits of (s * A + B) mod 2^32]; LUT from FILE (.npy), else Gaussian
                       quantiles in a random order (hfit.py fits it Lloyd-style)
Modes:
  eval SPECS KS LS NTILES TILESEED OUT.jsonl      one line per (spec, K, L): mse, f
  asearch FAMILY KS NTILES TILESEED rand:SEED:COUNT:BITS OUT.jsonl
                                                  FAMILY with A as {A} (e.g. h:{A}:0x2545F491:0x3F3F3F3F), random odd
                                                  A of BITS bits; m1i8 scored first on the same tiles (ratio)
One thread: OMP_NUM_THREADS=1. (w139)"""
import ctypes, json, os, sys, time
import numpy as np
from statistics import NormalDist

_d = os.path.dirname(os.path.abspath(__file__))
LIB = ctypes.CDLL(os.path.join(_d, "libvitL.so"))
LIB.viterbi_tiles_L.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]


def vit(x, cb, K, L, states=False):
    x = np.ascontiguousarray(x, dtype=np.float32); out = np.empty_like(x)
    st = np.empty(x.shape, dtype=np.int32) if states else None
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    LIB.viterbi_tiles_L(x.ctypes.data, out.ctypes.data, st.ctypes.data if states else None, x.shape[0], cb.ctypes.data, int(K), int(L))
    return (out, st) if states else out


def int8_grid(v):
    st = 3.5 / 127; q = np.clip(np.round(v / st), -127, 127) * st
    return q / q.std()


def std(v): v = v.astype(np.float64); return (v - v.mean()) / v.std()


def bytesum(h, mask):
    h = h & mask
    return sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.int64)


def hashv(A, B, L):
    s = np.arange(1 << L, dtype=np.uint64)
    return (s * np.uint64(A) + np.uint64(B)) & np.uint64(0xFFFFFFFF)


_gq = {}
def gauss_remap(S):
    """S's values -> Gaussian quantiles of their mid-CDF over the states"""
    vals, cnt = np.unique(S, return_counts=True)
    c = np.cumsum(cnt) / cnt.sum() - cnt / cnt.sum() / 2
    nd = NormalDist(); g = np.array([nd.inv_cdf(min(max(p, 1e-9), 1 - 1e-9)) for p in c])
    return g[np.searchsorted(vals, S)]


def codebook(spec, L=16):
    p = spec.split(":"); n = p[0]; I = lambda x: int(x, 0)
    if n in ("m1", "m1i8"):
        S = bytesum(hashv(0x83DCD12D, 0, L), np.uint64(0xFFFFFFFF))
        return std(S) if n == "m1" else int8_grid(std(S))
    if n == "h":
        return std(bytesum(hashv(I(p[1]), I(p[2]), L), np.uint64(I(p[3]))))
    if n == "hg":
        return int8_grid(std(gauss_remap(bytesum(hashv(I(p[1]), I(p[2]), L), np.uint64(I(p[3]))))))
    if n == "hc":
        v = std(bytesum(hashv(I(p[2]), I(p[3]), L), np.uint64(I(p[4])))); c = float(p[1])
        return int8_grid(std(v + c * v ** 3))
    if n == "hyb":
        A, B, IB = I(p[1]), I(p[2]), int(p[3])
        idx = (hashv(A, B, L) >> np.uint64(32 - IB)).astype(np.int64)
        if len(p) > 4: lut = np.load(":".join(p[4:]))
        else: lut = gauss_lut(IB)
        return lut[idx].astype(np.float64)
    raise SystemExit(f"unknown codebook {spec}")


def gauss_lut(IB, seed=1):
    """2^IB Gaussian quantiles in a random order (seed): a hash's neighbouring states land on near indices, so an ordered
    table would give them near values"""
    nd = NormalDist(); g = np.array([nd.inv_cdf((e + 0.5) / (1 << IB)) for e in range(1 << IB)])
    return g[np.random.default_rng(seed).permutation(1 << IB)]


FG = (0.9, 0.95, 1.0, 1.05, 1.1)


def mse_fit(x, cb, K, L, grid=FG):
    """min over scale f of mean((vit(x / f) f - x)^2): the grid, then a parabola through the best three"""
    m = [float(((vit(x / f, cb, K, L) * f - x) ** 2).mean()) for f in grid]
    i = int(np.argmin(m))
    if 0 < i < len(grid) - 1:
        a, b, c = np.polyfit(grid[i - 1:i + 2], m[i - 1:i + 2], 2)
        if a > 0:
            fs = -b / (2 * a)
            if grid[i - 1] < fs < grid[i + 1]:
                return float(((vit(x / fs, cb, K, L) * fs - x) ** 2).mean()), float(fs)
    return m[i], grid[i]


def tiles(n, seed): return np.random.default_rng(seed).standard_normal((n, 256)).astype(np.float32)


def main():
    mode = sys.argv[1]
    if mode == "eval":
        specs, KS, LS = sys.argv[2].split(","), [int(k) for k in sys.argv[3].split(",")], [int(l) for l in sys.argv[4].split(",")]
        n, ts, out = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
        x = tiles(n, ts)
        for L in LS:
            for spec in specs:
                cb = codebook(spec, L)
                for K in KS:
                    t0 = time.time(); m, f = mse_fit(x, cb, K, L)
                    o = {"spec": spec, "K": K, "L": L, "mse": m, "f": f, "ntiles": n, "tileseed": ts, "sec": round(time.time() - t0, 1)}
                    print(json.dumps(o), flush=True)
                    with open(out, "a") as fh: fh.write(json.dumps(o) + "\n")
    elif mode == "asearch":
        fam, KS, n, ts, rs, out = sys.argv[2], [int(k) for k in sys.argv[3].split(",")], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
        _, sd, cnt, bits = rs.split(":")
        A = (np.random.default_rng(int(sd)).integers(1 << (int(bits) - 1), 1 << int(bits), int(cnt)) | 1).tolist()
        x = tiles(n, ts)
        ref = {K: mse_fit(x, codebook("m1i8"), K, 16)[0] for K in KS}
        for a in A:
            spec = fam.replace("{A}", hex(a)); cb = codebook(spec)
            for K in KS:
                m, f = mse_fit(x, cb, K, 16)
                o = {"spec": spec, "A": a, "K": K, "mse": m, "f": f, "ratio": m / ref[K], "ntiles": n, "tileseed": ts}
                with open(out, "a") as fh: fh.write(json.dumps(o) + "\n")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
