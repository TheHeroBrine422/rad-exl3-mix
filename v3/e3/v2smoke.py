#!/usr/bin/env python3
"""v2smoke.py -- the TINY viterbi2 check. Must finish in < 30 s (target ~10-20 s, 4 threads).

Confirms, in one shot:
  1. both backends load and run (the search works at all),
  2. avx2 is bit-identical to scalar (and to the legacy libviterbi.so when present),
  3. the DP is sane: trellis MSE is better than plain round-to-nearest on a random grid,
  4. speed is sane: > 50 tiles/s per backend at 4 threads (a broken/serial build would be
     ~10-40 tiles/s; this check is what "not going to take hours" means).

Usage: python3 v2smoke.py [threads]     (default 4; the .so's must sit next to this file --
built in the judge image; on a dev box: see the build lines in V2.md / the Dockerfile).
"""
import ctypes, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
NTHR = int(sys.argv[1]) if len(sys.argv) > 1 else 4
os.environ["OMP_NUM_THREADS"] = str(NTHR)
P = ctypes.POINTER(ctypes.c_float)
T0 = time.time()
fails = []


def load(name, fn, v2=False):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    f = getattr(lib, fn)
    f.argtypes = [P, P, ctypes.c_int, P, ctypes.c_int] + ([ctypes.c_int] if v2 else [])
    return f


# legacy (viterbi.c): viterbi_tiles / viterbi_tiles_tb      v2: +V (+ntry for tb)
ref = load("libviterbi.so", "viterbi_tiles")
sc = load("libviterbi2.so", "viterbi2_tiles", v2=True)
ax = load("libviterbi2_avx2.so", "viterbi2_tiles", v2=True)
if not (sc and ax):
    sys.exit(f"V2 SMOKE FAIL: missing libviterbi2.so / libviterbi2_avx2.so in {HERE}")
scTB = load("libviterbi2.so", "viterbi2_tiles_tb", v2=True)
axTB = load("libviterbi2_avx2.so", "viterbi2_tiles_tb", v2=True)
refTB = load("libviterbi.so", "viterbi_tiles_tb")

import numpy as np
rng = np.random.default_rng(11)
NT = 128
x = (rng.standard_normal((NT, 256)) * 0.35).astype(np.float32)
cb1 = (rng.standard_normal(65536)).astype(np.float32)
cb2 = (rng.standard_normal(2 * 65536)).astype(np.float32)
o1 = np.zeros((NT, 256), np.float32)
o2 = np.zeros_like(o1)


def t(fn, *a):
    t0 = time.time()
    fn(*a)
    return time.time() - t0


def rate(dt, runs):
    return NT * runs / dt


# 1+2. free run: bit-exact across backends (+ legacy when present)
for K in (2, 4):
    dt_s = t(sc, x.ctypes.data_as(P), o1.ctypes.data_as(P), NT, cb1.ctypes.data_as(P), K, 1)
    refS = o1.copy()
    dt_a = t(ax, x.ctypes.data_as(P), o2.ctypes.data_as(P), NT, cb1.ctypes.data_as(P), K, 1)
    if not np.array_equal(refS, o2):
        fails.append(f"free K={K}: avx2 != scalar")
    if ref:
        t(ref, x.ctypes.data_as(P), o2.ctypes.data_as(P), NT, cb1.ctypes.data_as(P), K)
        if not np.array_equal(refS, o2):
            fails.append(f"free K={K}: legacy != scalar")
    rs, ra = rate(dt_s, 1), rate(dt_a, 1)
    print(f"  free K={K}: scalar {rs:.0f} tiles/s, avx2 {ra:.0f} tiles/s"
          + (f" (legacy bit-exact)" if ref else ""))
    if min(rs, ra) < 50:
        fails.append(f"free K={K}: slow ({min(rs, ra):.0f} tiles/s < 50)")

# 2. tail-biting ntry=1: bit-exact across backends (+ legacy)
for K in (3,):
    dt_s = t(scTB, x.ctypes.data_as(P), o1.ctypes.data_as(P), NT, cb1.ctypes.data_as(P), K, 1, 1)
    refS = o1.copy()
    dt_a = t(axTB, x.ctypes.data_as(P), o2.ctypes.data_as(P), NT, cb1.ctypes.data_as(P), K, 1, 1)
    if not np.array_equal(refS, o2):
        fails.append(f"tb K={K}: avx2 != scalar")
    if refTB:
        t(refTB, x.ctypes.data_as(P), o2.ctypes.data_as(P), NT, cb1.ctypes.data_as(P), K, 1)
        if not np.array_equal(refS, o2):
            fails.append(f"tb K={K}: legacy != scalar")
    print(f"  tb   K={K} ntry=1: scalar {rate(dt_s, 2):.0f} tiles/s, avx2 {rate(dt_a, 2):.0f} tiles/s"
          + (f" (legacy bit-exact)" if refTB else ""))

# 2. V=2 (HYB shape): bit-exact across backends
t(sc, x.ctypes.data_as(P), o1.ctypes.data_as(P), NT, cb2.ctypes.data_as(P), 4, 2)
refS = o1.copy()
t(ax, x.ctypes.data_as(P), o2.ctypes.data_as(P), NT, cb2.ctypes.data_as(P), 4, 2)
if not np.array_equal(refS, o2):
    fails.append("V=2 K=4: avx2 != scalar")
print("  V=2  K=4: scalar==avx2 bit-exact")

# 3. DP sanity: the trellis (K=4 = 4 bits/value) must beat SAME-RATE scalar quantization:
# nearest-neighbor to a 16-level grid from x's mid-quantiles. (NN to the full 65536-value
# codebook is NOT the right comparison -- that spends 16 bits/value, not 4.) A broken DP
# (constant state, etc.) would emit ~var(x) and fail this easily.
levels = np.quantile(x, (np.arange(16) + 0.5) / 16)
idx = np.searchsorted(levels, x)
nn16 = levels[np.clip(idx, 0, 15)]
mse_sq = float(np.mean((x - nn16) ** 2))
t(sc, x.ctypes.data_as(P), o1.ctypes.data_as(P), NT, cb1.ctypes.data_as(P), 4, 1)
mse_t = float(np.mean((x - o1) ** 2))
print(f"  sanity: trellis K=4 MSE {mse_t:.6f} < 16-level SQ {mse_sq:.6f}: {mse_t < mse_sq}")
if not mse_t < mse_sq:
    fails.append("DP sanity: trellis MSE not better than same-rate 16-level scalar quantization")

dt = time.time() - T0
if dt > 30:
    fails.append(f"smoke took {dt:.1f}s > 30s")
if fails:
    print("V2 SMOKE FAIL:"); [print("  -", f) for f in fails]
    sys.exit(1)
print(f"V2 SMOKE OK ({dt:.1f}s, {NTHR} threads, legacy ref {'yes' if ref else 'no'})")
