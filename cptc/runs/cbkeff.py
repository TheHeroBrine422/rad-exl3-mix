#!/usr/bin/env python3
"""cbkeff.py NAME K [BLOCKS] -- i.i.d. N(0,1) Keff of one codebook candidate (the w208 ALU-bench set) on the paper's
protocol (T = 256 tail-biting, L = 16), same final samples as keffcmp.py (seed 777, ntry KEFF_NTRY).

Fixed codebooks (mad, mul1_i8, int4grid, mad4, cptcmin) get a global scale search (the quantizer's scale search does the
same on real tensors). Table codebooks (lc1, cptc5, cptc6u, cptcst10) are fitted by Lloyd rounds (fresh samples, damping,
best held-out round), then the table is snapped to int8 (symmetric, scale = max|T| / 127) and refitted a few rounds.
Definitions follow v3/bench/cb208/README.md. Integer K only (fractional patterns need the pattern encoder).
"""
import os, sys, json, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("CPTC_E3DIR", "."))
import cptc, trellis

NAME, K = sys.argv[1], int(sys.argv[2])
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
NTRY = int(os.environ.get("KEFF_NTRY", 4))
S = np.arange(1 << 16, dtype=np.uint64)
A0, A1 = 0x9282, 0xA981          # the paper's unified pair (one pair for every K: fractional patterns need it)
M16 = np.uint64(0xFFFF)


def prod_hi(a, sh):
    return ((np.uint64(a) * S) & M16) >> np.uint64(sh)


def keff(x, cb, ntry):
    q, _ = cptc.encode_rev(x, np.ascontiguousarray(cb, dtype=np.float32), K, ntry)
    return cptc.keff_of(x, q)


def bytesum(h):
    h = h.astype(np.uint64)
    return sum(((h >> np.uint64(8 * i)) & np.uint64(0xFF)) for i in range(4)).astype(np.float64)


# ---- fixed codebooks: value per state (scale searched below) ----
def fixed_cb(name):
    if name in ("mad", "mul1_i8", "int4grid"):
        return np.asarray(trellis.codebook(name), dtype=np.float64)
    if name == "mad4":
        h = (S * np.uint64(0xD0474B) + np.uint64(0x2545F491)) & np.uint64(0xFFFFFFFF)
        n = ((h >> np.uint64(28)) & np.uint64(7)) + ((h >> np.uint64(24)) & np.uint64(7)) + ((h >> np.uint64(20)) & np.uint64(1))
        return n.astype(np.float64) - 8.0
    if name in ("madpop4", "madpop31"):   # popcount of mad-hash high bits: binomial(15) = 16 levels / binomial(31)
        h = (S * np.uint64(0xD0474B) + np.uint64(0x2545F491)) & np.uint64(0xFFFFFFFF)
        bits = (h >> np.uint64(17)) & np.uint64(0x7FFF) if name == "madpop4" else h >> np.uint64(1)
        n = np.zeros_like(bits)
        for b in range(31):
            n += (bits >> np.uint64(b)) & np.uint64(1)
        return n.astype(np.float64) - (7.5 if name == "madpop4" else 15.5)   # centred (11-21 fix: was uncentred)
    if name == "cptcmin":
        Q = math.ceil(K) + 1
        i = prod_hi(A0, 16 - Q) + prod_hi(A1, 16 - Q)
        return i.astype(np.float64) - (2 ** Q - 1)
    raise KeyError(name)


# ---- table codebooks: cell index per state, table size, mirror fold ----
def cells(name):
    if name == "lc1":
        return prod_hi(A1, 10).astype(np.int64), 64, False
    if name == "cptc5":
        return (prod_hi(A0, 11) + prod_hi(A1, 11)).astype(np.int64), 63, False
    if name == "cptc6u":
        return (prod_hi(A0, 10) + prod_hi(A1, 10)).astype(np.int64), 127, True
    if name in ("madlut", "madlutw", "madpoplut", "mul1lut"):   # hash index -> fitted int8 table
        if name == "mul1lut":
            h = (S * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
            return bytesum(h).astype(np.int64), 1021, False
        h = (S * np.uint64(0xD0474B) + np.uint64(0x2545F491)) & np.uint64(0xFFFFFFFF)
        if name == "madpoplut":
            b = (h >> np.uint64(17)) & np.uint64(0x7FFF); n = np.zeros_like(b)
            for i in range(15): n += (b >> np.uint64(i)) & np.uint64(1)
            return n.astype(np.int64), 16, False
        m = np.uint64(0x3F3F3F3F if name == "madlut" else 0x7F7F7F7F)
        return bytesum(h & m).astype(np.int64), (253 if name == "madlut" else 509), False
    if name.startswith("cptc1m"):   # ONE 24-bit multiply, two 6-bit fields (bits 10-15 + 26-31) summed: CPTC-like index
        C = int(name.split(":")[1], 16) if ":" in name else 0x9E3779
        p = (S * np.uint64(C)) & np.uint64(0xFFFFFFFF)
        return (((p >> np.uint64(10)) & np.uint64(63)) + (p >> np.uint64(26))).astype(np.int64), 127, False
    if name == "cptc7":   # unified pair, 7-bit fields: 255-entry table (wider alphabet for K5-6)
        return (prod_hi(A0, 9) + prod_hi(A1, 9)).astype(np.int64), 255, False
    if name == "cptcst10":
        return (S & np.uint64(1023)).astype(np.int64), 1024, False
    raise KeyError(name)


def gauss_init(c, n):
    """Conditional N(0,1) mean of each cell's probability interval, cells ordered by index."""
    from math import erf, sqrt, exp, pi
    h = np.bincount(c, minlength=n).astype(np.float64); G = np.cumsum(h) / h.sum()
    lo = np.concatenate([[0.0], G[:-1]]); hi = G
    def ppf(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        a, b = -10.0, 10.0
        for _ in range(80):
            m = 0.5 * (a + b)
            if 0.5 * (1 + erf(m / sqrt(2))) < p: a = m
            else: b = m
        return 0.5 * (a + b)
    phi = lambda z: exp(-z * z / 2) / sqrt(2 * pi)
    T = np.zeros(n)
    for j in range(n):
        if hi[j] - lo[j] <= 0: continue
        T[j] = (phi(ppf(lo[j])) - phi(ppf(hi[j]))) / (hi[j] - lo[j])
    return T


def expand(T, c, fold):
    if fold:   # 64 stored entries, i < 64 -> T[i], else -T[126 - i]
        j = np.where(c < 64, c, 126 - c); v = T[j]; return np.where(c < 64, v, -v)
    return T[c]


def centroid(T, c_state, st, x, n, fold):
    ci = c_state[st.astype(np.int64)].ravel(); xv = x.ravel().astype(np.float64)
    m = np.bincount(ci, weights=xv, minlength=n); k = np.bincount(ci, minlength=n).astype(np.float64)
    if fold:
        num = m[:63] - m[126:63:-1]; den = k[:63] + k[126:63:-1]
        Tn = T.copy(); Tn[:63] = np.where(den > 0, num / np.maximum(den, 1), T[:63]); Tn[63] = 0.0
        return Tn
    return np.where(k > 0, m / np.maximum(k, 1), T)


def snap8(T):
    s = np.max(np.abs(T)) / 127.0
    return np.round(T / s) * s


def fit_table(name, rng):
    c, n, fold = cells(name)
    T = gauss_init(c, n)
    if fold: T = T[:64].copy(); T[63] = 0.0
    xh = rng.standard_normal((1024, 256)).astype(np.float32)
    best = None
    for r in range(int(os.environ.get("CB_ROUNDS", 30))):
        xf = rng.standard_normal((1024, 256)).astype(np.float32)
        q, st = cptc.encode_rev(xf, expand(T, c, fold).astype(np.float32), K, 1, states_out=True)
        Tn = centroid(T, c, st, xf, n, fold)
        T = 0.5 * T + 0.5 * Tn if r >= 20 else Tn
        sc = keff(xh, expand(T, c, fold), 1)
        if best is None or sc > best[0]: best = (sc, T.copy())
    T = best[1]
    for r in range(6):   # int8 storage: snap, refit, snap
        T = snap8(T)
        xf = rng.standard_normal((1024, 256)).astype(np.float32)
        q, st = cptc.encode_rev(xf, expand(T, c, fold).astype(np.float32), K, 1, states_out=True)
        T = centroid(T, c, st, xf, n, fold)
    T = snap8(T)
    return expand(T, c, fold), {"table_entries": n if not fold else 64, "presnap_keff_heldout": round(best[0], 4)}


rng = np.random.default_rng(1000 + K)
extra = {}
if NAME in ("lc1", "cptc5", "cptc6u", "cptcst10", "madlut", "madlutw", "madpoplut", "mul1lut", "cptc7") or NAME.startswith("cptc1m"):
    cb, extra = fit_table(NAME, rng)
else:
    v = fixed_cb(NAME); v = v / np.sqrt(np.mean(v ** 2))
    xs = rng.standard_normal((512, 256)).astype(np.float32)
    grid = np.arange(0.40, 1.61, 0.1)
    sc = [keff(xs, v * g, 1) for g in grid]
    g0 = grid[int(np.argmax(sc))]
    fine = np.arange(g0 - 0.08, g0 + 0.081, 0.02)
    sc2 = [keff(xs, v * g, 1) for g in fine]
    g = float(fine[int(np.argmax(sc2))]); cb = v * g; extra = {"scale": round(g, 3)}
x = np.random.default_rng(777).standard_normal((NB, 256)).astype(np.float32)
out = {"name": NAME, "K": K, "blocks": NB, "ntry": NTRY, "keff": round(keff(x, cb, NTRY), 4), **extra}
print(json.dumps(out), flush=True)
