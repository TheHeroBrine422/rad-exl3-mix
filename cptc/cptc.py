#!/usr/bin/env python3
"""cptc.py -- CPTC (Computed Permutation Trellis Coding, arXiv 2611.22019) CPU reference.

Reference implementation for the C1/C2 verification workstream (local/TODO.md). The tail-biting
bitshift-trellis Viterbi is the workspace's EXL3 encoder (v3/e3/viterbi.c, `viterbi_tiles_tb*`):
CPTC's forward 16-bit window q_t = [low K: code_t | high 16-K: first 16-K stream bits AFTER code_t]
is bit-identical to the EXL3 state-after-shift of the REVERSED value sequence (low K: code | high
16-K: last 16-K bits BEFORE it, in reversed order), so

    CPTC-encode(x)  ==  exl3-tb-viterbi(reverse(x)) with the codebook function
                        cb[s] = CPTC_decode(s)  (Algorithm 1) precomputed over all 2^16 states.

The 65536-entry cb is exact for the encoder and for all checks; the SERVING decoder is the computed
two-multiply form (Algorithm 1) with the 64-byte table, not this LUT.

Protocol fidelity to the paper (assumptions noted):
  - Lmax = 16, Q = 6, T = 256, coefficients = Table 6 L=16 row (per K), s = 0.
  - Table generation (Sec 3.3 / A.2): Eq (2) conditional-mean init x (1+2/n)(1-4^-K), n = L-K+1;
    30 Lloyd rounds + 20 damped rounds (damping 0.5 -- NOT stated in the paper, assumption) at
    2^20 samples (paper: 2^22; scaled for speed), best round by held-out Keff; 16 E2M5-snap Lloyd
    rounds + final single-entry grid hill-climb.
  - Tail-biting two-pass with ntry = 16 rolls (viterbi.c; paper: "QTIP construction unchanged").

Usage:
  python3 cptc.py prop1                 # Proposition 1: scalar formats reproduced exactly
  python3 cptc.py table K                # generate the E2M5 table for K (Gaussian target), save
  python3 cptc.py keff K [--blocks N]    # Keff on i.i.d. standard normal, Table 2 target
  python3 cptc.py check K                # Algorithm 2 bit-exact cross-check on random blocks
Targets (Table 2, L=16, E2M5, Gaussian): K=1..6 -> 0.950 1.948 2.943 3.934 4.913 5.802
"""
import ctypes, json, math, os, sys, time
import numpy as np
from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.environ.get("CPTC_LIBDIR", os.path.join(os.path.dirname(HERE), "v3", "e3"))
LIB = ctypes.CDLL(os.path.join(E3, "libviterbi.so"))
LIBD = ctypes.CDLL(os.path.join(E3, "libviterbi_d.so"))  # double-precision build (exactness checks)
LIB.viterbi_tiles_tb.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_int]
LIB.viterbi_tiles_tb_st.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
LIBD.viterbi_tiles_tb.argtypes = LIB.viterbi_tiles_tb.argtypes
LIBD.viterbi_tiles_tb_st.argtypes = LIB.viterbi_tiles_tb_st.argtypes
ND = NormalDist()
def _cdf(p):
    p = np.asarray(p, dtype=np.float64)
    return np.array([ND.cdf(float(t)) for t in p.ravel()]).reshape(p.shape)

PHI, PHIP = _cdf, lambda x: np.exp(-np.asarray(x, dtype=np.float64) ** 2 / 2) / math.sqrt(2 * math.pi)
P16 = 1 << 16
Q = 6
LMAX = 16
TBL = os.path.join(HERE, "tables")

# Table 6, L = 16 row (a0, a1), per K
COEF = {1: (0x9097, 0x97AE), 2: (0x7AE6, 0xC8B3), 3: (0x7474, 0xAB1D),
        4: (0x6A78, 0x85CF), 5: (0x1C04, 0x3C01), 6: (0x0AEE, 0xD337)}
TARGET_KEFF = {1: 0.950, 2: 1.948, 3: 2.943, 4: 3.934, 5: 4.913, 6: 5.802}


def ctz(v):
    return (v & -v).bit_length() - 1


def depth(a0, a1):
    return LMAX - ctz(a0 | a1)


def e2m5_pos():
    """E2M5 non-negative magnitudes (paper Sec 5.1): sign + 2 exponent bits (bias 1) + 5 mantissa bits, subnormals,
    no infinities, all-ones magnitude = NaN. Exponent field 0 = subnormal m/32 (0 .. 31/32); fields 1..3 = normal
    2^(e-1) (1 + m/32), minus the NaN (e=3, m=31) -> steps 1/32 below 2, 1/16 in [2, 4), 1/8 in [4, 7.75]:
    127 magnitudes incl. 0 = 253 signed values.
    (Fixed 2026-11-20: the first version had subnormals m/64, i.e. no values in (0.484, 1.0), which with the
    missing storage scale below cost ~0.55 bits at K=4.)"""
    g = [m / 32 for m in range(32)]
    for e in (1, 2, 3):
        g += [2 ** (e - 1) * (1 + m / 32) for m in range(32) if not (e == 3 and m == 31)]
    g = np.array(sorted(set(g)))
    assert len(g) == 127 and g[-1] == 7.75
    return g


def storage_scale(grid=None):
    """Appendix C.3 rule: the largest 3-mantissa-bit float c such that c x (max entry of the K=4 Eq (2) starting
    table) does not saturate the type. Stored value = round(c v), decoded as stored / c. E2M5 -> 2 (paper)."""
    grid = e2m5_pos() if grid is None else grid
    vmax = float(np.max(np.abs(table_init_gauss(*COEF[4], 4) / ((1 + 2 / 13) * (1 - 4 ** -4)))))
    cands = sorted({m / 8 * 2.0 ** e for m in range(8, 16) for e in range(-4, 6)})
    return max(c for c in cands if c * vmax <= grid[-1])


def cell_index(a0, a1, states, shift=0):
    """Algorithm 1 lines 2-4, vectorized: states uint32 [n] -> i in [0, 2^(Q+1)-2=126]."""
    x0 = np.uint64(a0) * states.astype(np.uint64)
    x1 = np.uint64(a1) * states.astype(np.uint64)
    sh = LMAX - Q + shift
    return (((x0 & (P16 - 1)) >> sh) + ((x1 & (P16 - 1)) >> sh)).astype(np.int64)


def fold(Tlower, i):
    """Symmetric-storage lookup (Algorithm 1 lines 6-10): i<64 -> Tlower[i], else -Tlower[126-i]."""
    j = np.where(i < 64, i, 126 - i)
    v = Tlower[j]
    return np.where(i < 64, v, -v)


def decode_table(a0, a1, Tlower, shift=0):
    """cb[s] = CPTC_decode(s) for all 2^16 states (the 65536-entry EXL3-shaped codebook)."""
    s = np.arange(P16, dtype=np.uint32)
    return fold(Tlower, cell_index(a0, a1, s, shift)).astype(np.float64)


def encode_rev(x, cb, K, ntry=16, states_out=False, double=False):
    """CPTC encode via the reversed EXL3 tail-biting Viterbi. x [nt][256] -> q [nt][256] (original order).
    double=True uses the f64 build (exactness checks; production parity is f32)."""
    n, n2 = x.shape
    assert n2 == 256
    dt = np.float64 if double else np.float32
    assert x.dtype == dt
    cb = np.ascontiguousarray(cb, dtype=dt)
    xr = x[:, ::-1].copy()
    q = np.empty_like(xr)
    st = np.zeros((n, 256), dtype=np.uint16)
    lib = LIBD if double else LIB
    f = lib.viterbi_tiles_tb_st if states_out else lib.viterbi_tiles_tb
    if states_out:
        f(xr.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p),
          st.ctypes.data_as(ctypes.c_void_p), n, cb.ctypes.data_as(ctypes.c_void_p), K, ntry)
    else:
        f(xr.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p),
          n, cb.ctypes.data_as(ctypes.c_void_p), K, ntry)
    return q[:, ::-1], (st[:, ::-1] if states_out else None)


def algo2_decode(codes, K, a0, a1, Tlower, shift=0, n=256):
    """CPTC's own decode, straight from the packed stream (Algorithm 2). codes [nt][256] (K bits each,
    bit 0 = LSB of the stream). bits[t, iK+k] = bit k of codes[t, i]; q[t, i] = that window little-endian."""
    nt = codes.shape[0]
    bits = (((codes.astype(np.uint64)[:, :, None] >> np.arange(K, dtype=np.uint64)) & 1)
            .astype(np.uint8).transpose(0, 2, 1).reshape(nt, 256 * K))
    bp = np.concatenate([bits, bits[:, : LMAX - K]], axis=1)  # tail-biting wrap (line 1)
    q = np.zeros((nt, 256), dtype=np.uint32)
    for k in range(LMAX):
        q |= (bp[:, k::K][:, :256].astype(np.uint32)) << k
    return fold(Tlower, cell_index(a0, a1, q, shift))


def states_to_codes(st_rev, K):
    """EXL3-convention states (after-shift, reversed domain) -> original-order K-bit codes.
    low K bits of each state = that position's code."""
    return (st_rev & ((1 << K) - 1)).astype(np.uint64)


def cond_mean_gauss(a, b):
    """E[X | a <= X < b] for X ~ N(0,1); empty/degenerate intervals -> 0."""
    m = np.zeros(len(a))
    ok = b > a
    u, v = a[ok], b[ok]
    m[ok] = (PHIP(u) - PHIP(v)) / (PHI(v) - PHI(u))
    return m


def inv_cdf(p):
    out = np.empty_like(p, dtype=np.float64)
    for idx, t in np.ndenumerate(p):
        t = float(t)
        out[idx] = -np.inf if t <= 0.0 else (np.inf if t >= 1.0 else ND.inv_cdf(t))
    return out


def table_init_gauss(a0, a1, K):
    """Eq (2) conditional-mean table (full 127) rescaled by (1+2/n)(1-4^-K), n = L-K+1."""
    s = np.arange(P16, dtype=np.uint32)
    h = np.bincount(cell_index(a0, a1, s, 0), minlength=127)
    G = np.cumsum(h) / h.sum()  # G(i); G(-1) = 0
    u = np.concatenate([[0.0], G[:-1]])  # G(i-1), G(-1) = 0
    v = G                                # G(i); G(126) = 1.0
    Tfull = cond_mean_gauss(inv_cdf(u), inv_cdf(v))
    n = LMAX - K + 1
    Tfull *= (1 + 2 / n) * (1 - 4 ** -K)
    # mirror-symmetric signed table: Tlower[j] = Tfull[j] (j=0..63, negative tail .. center 0),
    # upper half derived by fold as -Tlower[126-i]
    Tlower = np.zeros(64)
    Tlower[:63] = (Tfull[:63] - Tfull[126:63:-1]) / 2
    return Tlower


def lloyd_round(Tlower, a0, a1, K, x, ntry=16):
    """One Lloyd round: encode all blocks, per-position cell, pooled cell means (mirror fold).
    Returns (Tlower_new, cells, q). x [nt][256]."""
    cb = decode_table(a0, a1, Tlower).astype(np.float32)
    q, st_rev = encode_rev(x, cb, K, ntry, states_out=True)
    # states of the ORIGINAL domain: reverse back (encode_rev already reversed the output)
    st = st_rev
    i = cell_index(a0, a1, st, 0)
    # pooled mirror-symmetric centroid of the pair (j, 126-j), j <= 62: (sum_j - sum_{126-j}) / (n_j + n_{126-j});
    # center cell 63 = 0 (the exact fold forces it). (Fixed 11-20: was the unweighted average of the two means.)
    m = np.bincount(i.ravel(), weights=x.ravel().astype(np.float64), minlength=127)
    c = np.bincount(i.ravel(), minlength=127).astype(np.float64)
    num = m[:63] - m[126:63:-1]; den = c[:63] + c[126:63:-1]
    Tnew = np.zeros(64)
    Tnew[:63] = np.where(den > 0, num / np.maximum(den, 1), Tlower[:63])  # keep old where unvisited
    return Tnew, i, q


def snap_e2m5(Tlower, grid, c=None):
    """Store each signed entry as round(c v) on the E2M5 grid, decode stored / c (Appendix C.3; c = storage_scale())."""
    c = STORE_C if c is None else c
    a = np.abs(Tlower) * c
    idx = np.clip(np.searchsorted(grid, a), 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    m = np.where(np.abs(a - lo) <= np.abs(a - hi), lo, hi)
    return np.sign(Tlower) * m / c


def keff_of(x, q):
    D = float(np.mean((x - q) ** 2))
    return -math.log2(math.sqrt(D))  # -log4(D) = -log2(sqrt(D)), N(X)=1


def gen_table(K, samples=2 ** 22, holdout=2 ** 18, seed=1234, verbose=True, fit_ntry=1):
    """Full A.2 protocol (damping 0.5 assumed -- NOT stated in the paper; fresh 2^22 samples per
    round as published). Returns Tlower (64,) signed E2M5-snapped table."""
    a0, a1 = COEF[K]
    assert depth(a0, a1) == 16, f"coeff pair for K={K} is not depth 16"
    rng = np.random.default_rng(seed)
    nblk = samples // 256
    hrng = np.random.default_rng(seed + 1)
    xh = hrng.standard_normal(((holdout // 256) * 256, 256)).astype(np.float32)
    grid = e2m5_pos()
    T = table_init_gauss(a0, a1, K).astype(np.float64)
    best = None
    for r in range(50):
        damped = r >= 30
        xf = rng.standard_normal((nblk, 256)).astype(np.float32)  # fresh samples per round (paper)
        Tnew, _, _ = lloyd_round(T, a0, a1, K, xf, ntry=fit_ntry)
        T = 0.5 * T + 0.5 * Tnew if damped else Tnew
        if verbose and (r % 5 == 0 or r >= 30):
            print(f"  K={K} round {r:2d}{' (damped)' if damped else ''}", flush=True)
        qh, _ = encode_rev(xh, decode_table(a0, a1, T).astype(np.float32), K, fit_ntry)
        sc = keff_of(xh, qh)
        if best is None or sc > best[0]:
            best = (sc, T.copy())
    T = best[1]
    # E2M5 snap: 16 snap-Lloyd rounds (fresh samples per round)
    for r in range(16):
        xf = rng.standard_normal((nblk, 256)).astype(np.float32)
        T, _, _ = lloyd_round(T, a0, a1, K, xf, ntry=fit_ntry)
        T = snap_e2m5(T, grid)
    if verbose:
        qh, _ = encode_rev(xh, decode_table(a0, a1, T).astype(np.float32), K, fit_ntry)
        print(f"  K={K} after snap rounds: held-out Keff {keff_of(xh, qh):.4f} (pre-snap best {best[0]:.4f})", flush=True)
    # final pass: move single entries to a neighbouring grid value while the held-out error falls (paper A.2).
    # (Fixed 11-20: the first version tried grid[j +- 1] for table entry j -- an unrelated value -- once.)
    def score(Tt):
        qh2, _ = encode_rev(xh, decode_table(a0, a1, Tt).astype(np.float32), K, fit_ntry)
        return keff_of(xh, qh2)
    base = score(T)
    for sweep in range(8):
        moved = 0
        for j in range(63):                        # entry 63 = the center cell, fixed at 0
            g = int(np.argmin(np.abs(grid - abs(T[j]) * STORE_C)))
            sg = np.sign(T[j]) if T[j] != 0 else 1.0
            for gg in (g - 1, g + 1):
                if not (0 <= gg < len(grid)):
                    continue
                Tt = T.copy(); Tt[j] = sg * grid[gg] / STORE_C
                sc = score(Tt)
                if sc > base:
                    T, base, moved = Tt, sc, moved + 1
                    break
        if verbose:
            print(f"  K={K} hill-climb sweep {sweep}: {moved} moves, held-out Keff {base:.4f}", flush=True)
        if moved == 0:
            break
    os.makedirs(TBL, exist_ok=True)
    np.save(os.path.join(TBL, f"table-K{K}.npy"), T)
    np.save(os.path.join(TBL, f"table-K{K}-presnap.npy"), best[1])
    print(f"K={K}: saved {len(T)}-entry E2M5 table (best-round held-out Keff {best[0]:.4f}, "
          f"post-snap {base:.4f})")
    return T


def keff(K, blocks=16384, seed=777, ntry=16):
    """Keff on fresh i.i.d. standard-normal blocks with the generated table (Table 2 target)."""
    T = np.load(os.path.join(TBL, f"table-K{K}.npy"))
    a0, a1 = COEF[K]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((blocks, 256)).astype(np.float32)
    cb = decode_table(a0, a1, T).astype(np.float32)
    t0 = time.time()
    q, _ = encode_rev(x, cb, K, ntry)
    dt = time.time() - t0
    D = float(np.mean((x - q) ** 2))
    k = keff_of(x, q)
    print(f"K={K}: Keff {k:.4f} (target {TARGET_KEFF[K]}, spread ~0.0005/seed) "
          f"blocks={blocks} {dt:.1f}s  gap {k - TARGET_KEFF[K]:+.4f}")
    return k


def prop1(K=4):
    """Proposition 1: a0=0, a1=2^(16-K), s=Q-K reproduces any 2^K codebook exactly (f64 DP -- the f32
    build shows f32-accumulation flips in this degenerate, fully-separable case)."""
    a0, a1 = 0, 1 << (16 - K)
    shift = Q - K
    rng = np.random.default_rng(0)
    cbk = np.sort(rng.standard_normal(2 ** K))  # arbitrary codebook (Lloyd-like)
    Tlower = np.zeros(64)
    Tlower[:2 ** K] = cbk  # only cells 0..2^K-1 are reachable
    x = rng.standard_normal((64, 256)).astype(np.float64)
    cb = decode_table(a0, a1, Tlower, shift)
    # expected: per-value nearest codebook entry
    nearest = cbk[(np.abs(x[..., None] - cbk[None, None, :]).argmin(-1))]
    q, _ = encode_rev(x, cb, K, 16, double=True)
    bad = int(np.sum(q != nearest))
    print(f"Prop 1 K={K}: mismatches {bad}/{q.size}  "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def check(K, blocks=32, seed=5):
    """Bit-exact cross-check: encoder output vs CPTC's own Algorithm 2 window decode."""
    T = np.load(os.path.join(TBL, f"table-K{K}.npy"))
    a0, a1 = COEF[K]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((blocks, 256)).astype(np.float32)
    cb = decode_table(a0, a1, T).astype(np.float32)
    q, states = encode_rev(x, cb, K, 16, states_out=True)
    codes = states_to_codes(states, K)
    ref = algo2_decode(codes, K, a0, a1, T)
    bad = int(np.sum(q != ref))
    # and against the computed 2-multiply decode of the states (no LUT):
    i = cell_index(a0, a1, states, 0)
    comp = fold(T, i).astype(np.float32)
    bad2 = int(np.sum(comp != q))
    print(f"check K={K}: algo2 mismatches {bad}/{q.size}, computed-vs-lut {bad2}/{q.size}  "
          f"{'PASS' if bad == 0 and bad2 == 0 else 'FAIL'}")
    return bad == 0 and bad2 == 0


STORE_C = storage_scale()
assert STORE_C == 2.0, STORE_C   # the paper's E2M5 scale (Appendix C.3)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prop1":
        prop1(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif cmd == "table":   # table K [samples_per_round]
        gen_table(int(sys.argv[2]), samples=int(sys.argv[3]) if len(sys.argv) > 3 else 2 ** 22)
    elif cmd == "keff":
        keff(int(sys.argv[2]),
             blocks=int(sys.argv[3]) if len(sys.argv) > 3 else 16384,
             ntry=int(sys.argv[4]) if len(sys.argv) > 4 else 16)
    elif cmd == "check":
        check(int(sys.argv[2]))
