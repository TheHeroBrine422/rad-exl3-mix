#!/usr/bin/env python3
"""trellisL.py -- w71 (#113): the L-bit-window trellis (beamL.c) from Python, and the mul1 family generalised to L bits.

Codebook for an L-bit state s (L = 16..20): exactly today's mul1 decode with a wider window,
    h = (s * 0x83DCD12D) mod 2^32,  bs = byte0(h) + byte1(h) + byte2(h) + byte3(h)  (0..1020),  value = vtab[bs],
where vtab is today's value of each byte sum (mul1: the affine map; mul1_i8: the affine map then the int8 grid; any
codebook of trellis2 / settle64 that is a function of bs). The value set and the decode arithmetic are unchanged;
only the window mask (0xffff -> 2^L - 1) differs. vtab is read off the 16-bit table, so L = 16 is today's table bit
for bit. The C search reads a 2^L table of these values (4 MiB at L = 20), the encoder's cache of a computed code.

lib(): libbeamL.so next to this file or in $W71_LIB (built on the box: gcc -O3 -march=native -fopenmp -shared -fPIC).
search(tiles, cb, L, K, M, tb): M = 0 full Viterbi, else the M-algorithm; tb 0 free start, 1 exllamav3's tail-biting
heuristic (rolled free run -> wrap group -> constrained run)."""
import ctypes, os
import numpy as np

MUL = 0x83DCD12D
_lib = None


def lib():
    global _lib
    if _lib is None:
        p = os.environ.get("W71_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libbeamL.so")
        _lib = ctypes.CDLL(p)
        _lib.trellis_tiles.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p] + [ctypes.c_int] * 4
    return _lib


def bytesum(L):
    s = np.arange(1 << L, dtype=np.uint64)
    h = (s * MUL) & 0xFFFFFFFF
    return sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.int64)


def vtab(name):
    """the value of every byte sum 0..1020 under codebook `name`, by today's formula (trellis.codebook: mul1 = the byte
    sum standardised over the 65536 16-bit states, mul1_i8 = that on the 3.5/127 int8 grid; settle64: mul1x / mul1x_i8
    = the fork's fp16 decode). At L = 16 some extreme byte sums never occur, at L = 20 they do (a handful of states)."""
    b = np.arange(1021, dtype=np.float64); b16 = bytesum(16).astype(np.float64)
    if name in ("mul1", "mul1_i8"):
        v = (b - b16.mean()) / b16.std()
        if name == "mul1_i8": st = 3.5 / 127; v = np.clip(np.round(v / st), -127, 127) * st
        return v
    if name in ("mul1x", "mul1x_i8"):
        f16 = lambda u: float(np.frombuffer(np.uint16(u).tobytes(), np.float16)[0])
        v = (np.float16(1024 + b).astype(np.float64) * f16(0x1eee) + f16(0xc931)).astype(np.float16)
        if name == "mul1x": return v.astype(np.float64)
        s4 = np.float32(4.0) * np.float32(f16(0x1eee))
        q = np.clip(np.rint(v.astype(np.float32) * (np.float32(1.0) / s4)), -127, 127).astype(np.float32)
        return (q * s4).astype(np.float16).astype(np.float64)
    raise SystemExit(f"no L-bit form for codebook {name}")


def codebook_L(name, L, cb16=None):
    """the L-bit generalisation of mul1-family codebook `name` (float32, 2^L entries). cb16 (the 16-bit table the
    rest of the code uses) is checked to equal the L = 16 form exactly"""
    out = np.ascontiguousarray(vtab(name)[bytesum(L)], dtype=np.float32)
    if cb16 is not None:
        t16 = out if L == 16 else np.ascontiguousarray(vtab(name)[bytesum(16)], dtype=np.float32)
        if not np.array_equal(t16, np.asarray(cb16, dtype=np.float32)): raise SystemExit(f"{name}: L16 form != table")
    return out


def rand_table(L, name="mul1_i8", seed=7):
    """an iid N(0,1) table of 2^L values (seed 7, as hashdiag.py's "rand"), standardised, on mul1_i8's int8 grid
    (3.5 / 127) for i8 codebooks: the "ideal code" a hash could approach (no cheap decoder)"""
    v = np.random.default_rng(seed).standard_normal(1 << L); v = (v - v.mean()) / v.std()
    if name.endswith("i8"): st = 3.5 / 127; v = np.clip(np.round(v / st), -127, 127) * st
    return np.ascontiguousarray(v, dtype=np.float32)


def search(x, cb, L, K, M=0, tb=1, want_cost=False):
    """x [ntiles, 256] -> quantized values [ntiles, 256] float32 (and the per-tile search SSE)"""
    assert float(K).is_integer() and len(cb) == 1 << L
    x = np.ascontiguousarray(x, dtype=np.float32); out = np.empty_like(x)
    cost = np.empty(x.shape[0], dtype=np.float32)
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    lib().trellis_tiles(x.ctypes.data, out.ctypes.data, None, cost.ctypes.data, x.shape[0], cb.ctypes.data, L, int(K),
                        int(M), int(tb))
    return (out, cost) if want_cost else out
