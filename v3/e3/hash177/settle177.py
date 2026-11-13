#!/usr/bin/env python3
"""settle177.py LAYER TENSOR K OUT [ROWS] -- w71's settle71.py (#113: exllamav3's LDLQ order, transforms, g_scale search,
refit; the L-bit tail-biting Viterbi beamL.c; arms V16 / V17 / V18 ...) with #177's hash codebooks, by name in W71_CB:
  hash:A:B:MASK        S = bytesum(((s * A + B) mod 2^32) & MASK) of the L-bit state s (A < 2^24: one v_mad_u32_u24 for
                       any L <= 24), value = S standardised over the 16-bit states (an int8-grid integer S - 126 at MASK
                       0x3F3F3F3F); at L = 16 = hscreen's h:A:B:MASK
  hashg:A:B:MASK       the same S through its Gaussian remap (hscreen's hg, from the 16-bit CDF), int8 grid
  hashc:C:A:B:MASK     the same S through v + C v^3 (hscreen's hc), int8 grid
Other names (mul1_i8, ...) are settle71's own. Everything else is settle71.py unchanged (run with its env: W71_ARMS,
W71_LIB, W71_SCALES, SETTLE_CACHE, SETTLE_ORDER=rev, E3_*). (w139)"""
import os, sys
import numpy as np

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "code/settle/w71"), os.path.join(_h, "code/settle/w64"), os.path.join(_h, "code/settle"),
                os.path.join(_h, "code")]
import trellis2, trellisL as TL, settle71
from statistics import NormalDist


def _S(A, B, mask, L):
    s = np.arange(1 << L, dtype=np.uint64)
    h = (s * np.uint64(A) + np.uint64(B)) & np.uint64(0xFFFFFFFF) & np.uint64(mask)
    return sum(((h >> np.uint64(8 * i)) & np.uint64(0xFF)) for i in range(4)).astype(np.int64)


def _i8(v):
    st = 3.5 / 127; return np.clip(np.round(v / st), -127, 127) * st


def hash_vals(name, L):
    p = name.split(":"); I = lambda x: int(x, 0)
    kind = p[0]
    if kind == "hashc": c, (A, B, M) = float(p[1]), map(I, p[2:5])
    else: A, B, M = map(I, p[1:4])
    assert A < 1 << 24
    S16 = _S(A, B, M, 16).astype(np.float64); mu, sd = S16.mean(), S16.std()
    S = _S(A, B, M, L)
    if kind == "hash": return (S - mu) / sd
    if kind == "hashg":
        vals, cnt = np.unique(S16, return_counts=True)
        cdf = np.cumsum(cnt) / cnt.sum() - cnt / cnt.sum() / 2
        nd = NormalDist(); g = np.array([nd.inv_cdf(min(max(q, 1e-9), 1 - 1e-9)) for q in cdf])
        tab = np.zeros(int(S.max()) + 1); tab[vals] = g
        g16 = tab[S16.astype(np.int64)]; v = (tab[S] - g16.mean()) / g16.std()
        return _i8(v)
    if kind == "hashc":
        v16 = (S16 - mu) / sd; y16 = v16 + c * v16 ** 3
        v = (S - mu) / sd; return _i8((v + c * v ** 3 - y16.mean()) / y16.std())
    raise SystemExit(f"unknown hash codebook {name}")


_cb0, _cbL0 = trellis2.codebook, TL.codebook_L
def codebook(name):
    if name.split(":")[0] in ("hash", "hashg", "hashc"): return hash_vals(name, 16)
    return _cb0(name)
def codebook_L(name, L, cb16=None):
    if name.split(":")[0] in ("hash", "hashg", "hashc"):
        out = np.ascontiguousarray(hash_vals(name, L), dtype=np.float32)
        if cb16 is not None and not np.array_equal(np.ascontiguousarray(hash_vals(name, 16), dtype=np.float32),
                                                    np.asarray(cb16, dtype=np.float32)):
            raise SystemExit(f"{name}: L16 form != table")
        return out
    return _cbL0(name, L, cb16)
trellis2.codebook = codebook; TL.codebook_L = codebook_L

if __name__ == "__main__":
    settle71.main()
