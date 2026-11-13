#!/usr/bin/env python3
"""embref.py SRC CB OUT.rad [JOBS] -- (w144, #174-02) the dequantized reference of a trellis token table: every row of
SRC's model.language_model.embed_tokens (an exl3i importer source, embsrc.py) decoded by THE ORACLE (plugin
exl3import/common/exl3i_decode.c exl3i_decode_rows: what a load-time expansion would hold), rounded RNE to bf16, written
over OUT.rad's token_embd.weight in place (OUT must already hold a plain bf16 [248320, 5120] token_embd: copy a pack whose
embedding the recipe kept as the checkpoint's). Natural row order only. CB: mul1 | mul1i8 | mul1i4. Needs LIB =
libexl3iref.so (env EXL3IREF, default next to this file): gcc -O2 -ffp-contract=off -fPIC -shared exl3i_decode.c
exl3_decode.c. Parallel over 128-row blocks (fork; JOBS, default 12)."""
import ctypes, os, struct, sys, time
import numpy as np
from multiprocessing import Pool
from safetensors import safe_open

SRC, CBN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
JOBS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
CB = {"mul1": 0, "mul1i8": 1, "mul1i4": 2}[CBN]
LIB = os.environ.get("EXL3IREF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libexl3iref.so"))
V, E = 248320, 5120

def entry(path, want):                       # v3/yaqa/headslab.py's RAD2 directory reader
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

class View(ctypes.Structure):
    _fields_ = [("rows", ctypes.c_int64), ("cols", ctypes.c_int64), ("m", ctypes.c_int), ("cb", ctypes.c_int),
                ("record", ctypes.c_void_p), ("suh", ctypes.c_void_p), ("suh1", ctypes.c_void_p), ("svh", ctypes.c_void_p)]

P = "model.language_model.embed_tokens"
with safe_open(f"{SRC}/q-embed.safetensors", "np") as f:
    tr = f.get_tensor(f"{P}.trellis"); suh = np.ascontiguousarray(f.get_tensor(f"{P}.suh")); svh = np.ascontiguousarray(f.get_tensor(f"{P}.svh"))
kt, nt, m = tr.shape; assert (kt * 16, nt * 16) == (E, V), tr.shape
W = m // 2
REC = np.ascontiguousarray(tr.view(np.uint32).reshape(kt, nt, W).transpose(1, 0, 2))   # table row tn, tiles in k order
del tr
off, nbytes, shape = entry(OUT, "token_embd.weight")
assert tuple(shape) == (V, E) and nbytes == V * E * 2, (shape, nbytes)

def bf16(x):                                 # f32 -> bf16 bits, RNE (NaN kept a NaN)
    u = x.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r

def run(blocks):
    lib = ctypes.CDLL(LIB)
    v = View(V, E, m, CB, REC.ctypes.data, suh.ctypes.data, None, svh.ctypes.data)
    out = np.memmap(OUT, np.uint16, "r+", offset=off, shape=(V, E))
    buf = np.empty((128, E), np.float32)
    for b in blocks:
        rc = lib.exl3i_decode_rows(ctypes.byref(v), ctypes.c_int64(b * 128), ctypes.c_int64(128), buf.ctypes.data_as(ctypes.c_void_p))
        assert rc == 0, rc
        out[b * 128:(b + 1) * 128] = bf16(buf.reshape(-1)).reshape(128, E)
    out.flush()
    return len(blocks)

if __name__ == "__main__":
    t = time.time(); nb = V // 128
    parts = [list(range(i, nb, JOBS)) for i in range(JOBS)]
    with Pool(JOBS) as p: n = sum(p.map(run, parts))
    print(f"{OUT}: token_embd.weight <- oracle rows of {SRC} ({CBN}, m {m}), {n} blocks, {time.time() - t:.0f} s")
