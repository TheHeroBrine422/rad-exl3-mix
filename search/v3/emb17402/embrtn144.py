#!/usr/bin/env python3
"""embrtn144.py OUT.rad -- (w144, #174-02) the int8 per-row RTN reference of the token table (the simple host-RAM
embedding of the lead's 11-13 note): OUT.rad's own bf16 token_embd.weight (a copy of a pack that kept the checkpoint's
table) overwritten in place by bf16(q * s), s = bf16(absmax_row / 127), q = round(w / s) in [-127, 127] (int8rel144.py's
"int8-row"; the f32 product rounded RNE to bf16, what an int8 gather writing bf16 would produce). Served by embed_lookup,
it scores what that gather would. entry() / bf16() as embref.py."""
import struct, sys, time
import numpy as np

OUT = sys.argv[1]
V, E = 248320, 5120


def entry(path, want):                       # v3/yaqa/headslab.py's RAD2 directory reader (embref.py)
    f = open(path, "rb"); h = f.read(4096)
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 16 + 5 * 8)
    f.seek(str_off); S = f.read(str_bytes)
    es = 3 * 8 + 4 + 4 + 8 * 6 + 4 * 3 + 4 + 8 + 8 + 8 + 16
    f.seek(dir_off); D = f.read(es * dir_cnt)
    for i in range(dir_cnt):
        e = D[i * es:(i + 1) * es]
        name = struct.unpack_from("<Q", e, 0)[0]
        if S[name:S.index(b"\0", name)].decode() != want: continue
        rank = struct.unpack_from("<I", e, 28)[0]; shape = struct.unpack_from("<6q", e, 32)[:rank]
        off, nbytes = struct.unpack_from("<QQ", e, es - 32)
        return off, nbytes, shape
    raise KeyError(want)


def bf16(x):                                 # f32 -> bf16 bits, RNE (NaN kept a NaN)
    u = x.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r


t = time.time()
off, nbytes, shape = entry(OUT, "token_embd.weight")
assert tuple(shape) == (V, E) and nbytes == V * E * 2, (shape, nbytes)
T = np.memmap(OUT, np.uint16, "r+", offset=off, shape=(V, E))
err = ref = 0.0
for r0 in range(0, V, 4096):
    w = (T[r0:r0 + 4096].astype(np.uint32) << 16).view(np.float32)
    s = (bf16(np.abs(w).max(1, keepdims=True) / np.float32(127)).astype(np.uint32) << 16).view(np.float32)
    s = np.maximum(s, np.float32(1e-30))
    q = np.clip(np.rint(w / s), -127, 127).astype(np.float32)
    o = bf16(np.ascontiguousarray(q * s))
    d = (o.astype(np.uint32) << 16).view(np.float32) - w
    err += float((d.astype(np.float64) ** 2).sum()); ref += float((w.astype(np.float64) ** 2).sum())
    T[r0:r0 + 4096] = o
T.flush()
print(f"{OUT}: token_embd.weight <- int8 per-row RTN (bf16 out), rel {err / ref:.4e}, {time.time() - t:.0f} s")
