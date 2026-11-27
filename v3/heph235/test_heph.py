#!/usr/bin/env python3
"""test_heph.py -- checks heph.c: (1) DP == brute force (free and constrained passes) on tiny L / T / patterns, incl. 0-bit steps;
(2) tail-biting validity of full encodes (every transition, the wrap, codewords = cb[state ^ salt]); (3) all codewords on the E2M1
grid. Run: python3 -I test_heph.py (libheph.so built next to it)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import heph


def salt(pos, k, L): return 0 if k else (0x9E37 * ((pos & 15) + 1)) & ((1 << L) - 1)


def brute(z, w, kk, L, cb, c):
    """min over all state paths with valid transitions (and the seam constraint if c >= 0), by DFS enumeration."""
    T = len(z); best = [1e30]
    def rec(i, s, cost):
        if cost >= best[0]: return
        if i == T:
            if c < 0 or (s & ((1 << (L - kk[0])) - 1)) == c: best[0] = cost
            return
        k = kk[i]
        for b in range(1 << k):
            ns = ((s << k) | b) & ((1 << L) - 1) if k else s
            d = cb[ns ^ salt(i, k, L)] - z[i]; rec(i + 1, ns, cost + w[i] * d * d)
    for s0 in range(1 << L):
        if c >= 0 and (s0 >> kk[0]) != c: continue
        d = cb[s0 ^ salt(0, kk[0], L)] - z[0]; rec(1, s0, w[0] * d * d)
    return best[0]


def main():
    rng = np.random.default_rng(0)
    for K, L, T in [(1, 4, 8), (2, 5, 7), (3, 6, 6), (1.5, 5, 8), (2.5, 6, 6), (0.5, 4, 8)]:
        pat = heph.pattern(K); Kt = max(max(pat), 1)
        kk = np.array([pat[i % len(pat)] for i in range(T)], dtype=np.int32)
        cb = heph.codebook(L, Kt, 3)
        for trial in range(6):
            z = rng.choice(heph.FP4_TABLE, size=T).astype(np.float32); w = rng.random(T).astype(np.float32) + 0.1
            for c in (-1, int(rng.integers(0, 1 << (L - kk[0])))):
                st = np.empty(T, dtype=np.uint16)
                m = heph.LIB.heph_free_pass(z.ctypes.data, w.ctypes.data, kk.ctypes.data, T, L, cb.ctypes.data, c, st.ctypes.data)
                b = brute(z, w, kk, L, cb, c)
                assert abs(m - b) <= 1e-4 * max(1, b), (K, L, T, c, m, b)
        print(f"brute-force match K={K} L={L} T={T}")
    for K, L, T in [(1, 12, 64), (2, 12, 64), (3, 12, 64), (2, 16, 64), (1.5, 12, 64), (2.5, 12, 64)]:
        z = rng.choice(heph.FP4_TABLE, size=(40, T)).astype(np.float32); w = rng.random((40, T)).astype(np.float32) + 0.1
        out, st, cost, bad, cb, kk = heph.encode(z, w, K, L, T, 5, want_states=True)
        assert bad == 0
        for n in range(40):
            for i in range(T):
                k = int(kk[i]); prev = int(st[n, i - 1])      # i = 0 wraps to the last position
                assert (int(st[n, i]) >> k) == (prev & ((1 << (L - k)) - 1)) or (k == 0 and st[n, i] == prev), (K, L, i)
                assert out[n, i] == cb[int(st[n, i]) ^ salt(i, k, L)]
        assert set(np.unique(np.abs(out)).tolist()) <= set(heph.MAGS.tolist())
        print(f"tail-biting valid + on-grid K={K} L={L}")
    print("ALL OK")


if __name__ == "__main__": main()
