"""heph.py -- HEPH grid-aligned trellis encoder wrapper (w235): MagStrat codebook, shift patterns, ctypes binding of libheph.so,
MXFP4 exponent Huffman cost. The Viterbi itself is heph.c."""
import ctypes, heapq, os
from fractions import Fraction
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = ctypes.CDLL(os.environ.get("HEPH_LIB", os.path.join(HERE, "libheph.so")))
LIB.heph_encode.restype = ctypes.c_int
LIB.heph_encode.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 4
LIB.heph_free_pass.restype = ctypes.c_float
LIB.heph_free_pass.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

MAGS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def pattern(K):
    """bits per step for K = a/b (same rule as v3/e3/trellis.py pattern): b steps, a bits, spread evenly."""
    f = Fraction(K).limit_denominator(8); a, b = f.numerator, f.denominator
    return [((i + 1) * a) // b - (i * a) // b for i in range(b)]


def codebook(L, Kt, seed=1):
    """MagStrat table [2^L] float32: c(s) = sign[s] * M[perm_h[b]], h = s >> Kt, b = s & (2^Kt - 1); perm_h a per-history
    permutation of the 8 magnitudes (only its first 2^Kt entries are reachable when Kt < 3); sign random per state."""
    assert 1 <= Kt <= 3
    rng = np.random.default_rng(seed)
    NH = 1 << (L - Kt)
    perm = np.argsort(rng.random((NH, 8)), axis=1)
    sgn = rng.integers(0, 2, size=1 << L) * 2 - 1
    s = np.arange(1 << L)
    return (sgn * MAGS[perm[s >> Kt, s & ((1 << Kt) - 1)]]).astype(np.float32)


def encode(z, w, K, L, T, seed=1, want_states=False):
    """z, w [nseq, T] float32 (targets, branch weights) -> (codewords [nseq, T] float32, states, per-seq cost, #infeasible, cb, kk)."""
    z = np.ascontiguousarray(z, dtype=np.float32); w = np.ascontiguousarray(w, dtype=np.float32)
    pat = pattern(K); Kt = max(max(pat), 1)
    kk = np.ascontiguousarray([pat[i % len(pat)] for i in range(T)], dtype=np.int32)
    cb = codebook(L, Kt, seed)
    out = np.empty_like(z); cost = np.empty(z.shape[0], dtype=np.float32)
    st = np.empty(z.shape, dtype=np.uint16) if want_states else None
    bad = LIB.heph_encode(z.ctypes.data, w.ctypes.data, kk.ctypes.data, z.shape[0], T, L, cb.ctypes.data, out.ctypes.data,
                          st.ctypes.data if want_states else None, cost.ctypes.data)
    return out, st, cost, bad, cb, kk


def huffman_lengths(counts):
    """canonical Huffman code lengths for {symbol: count} (one symbol -> length 1)."""
    if len(counts) == 1: return {next(iter(counts)): 1}
    heap = [(c, i, [s]) for i, (s, c) in enumerate(counts.items())]; heapq.heapify(heap)
    depth = {s: 0 for s in counts}; n = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap); c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2: depth[s] += 1
        heapq.heappush(heap, (c1 + c2, n, s1 + s2)); n += 1
    return depth


def exponent_cost(E):
    """E [N, G] uint8 e8m0 bytes of one matrix (G groups of 32 along the input dim) -> dict of bits per WEIGHT:
    rows store their first byte raw (8 bits) and the deltas along the input dim with one canonical Huffman code per matrix."""
    E = E.astype(np.int16); N, G = E.shape
    d = (E[:, 1:] - E[:, :-1]).reshape(-1)
    vals, cnt = np.unique(d, return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals, cnt)}
    ln = huffman_lengths(counts); tot = sum(ln[v] * c for v, c in counts.items())
    p = cnt / cnt.sum(); ent = float(-(p * np.log2(p)).sum())
    nw = N * G * 32
    return {"huff_bpw": (tot + 8 * N) / nw, "delta_bits_mean": tot / d.size, "delta_entropy": ent,
            "frac_delta0": float((d == 0).mean()), "frac_delta_pm1": float((np.abs(d) <= 1).mean()), "first_byte_bpw": 8 * N / nw}
