#!/usr/bin/env python3
"""trellis.py LAYER TENSOR CODEBOOKS BITS [ROWS] [OUT.jsonl] -- E3 trellis check on a sample of output rows (CPU).

BITS may be fractional (V3 §6b): K = a/b runs a shift pattern of b steps summing to a bits, spread evenly
(2.5 = [2,3], 2.25 = [2,2,2,3], 0.5 = [0,1]; a 0-bit step keeps the state and reads the codebook at a salted index,
viterbi_tiles_pat). Integer K uses the same pattern code path (one-element pattern).

Block LDLQ (QuIP#/EXL3 form): 16-column blocks of the rotated weight, each block's 16x16 tiles quantized jointly by
a bitshift-trellis Viterbi (viterbi.c), the block's error fed forward through the Hessian:
  W[:, b:] -= (W_blk - Q_blk) @ inv(U_blk,blk) @ U_blk,b:,   U = chol(H^-1) upper.
Rows are independent given H, so ROWS (default 256) random rows give an unbiased estimate of the tensor's relative
proxy error rel = tr(dW H dW^T) / tr(W H W^T) -- the hproxy metric, comparable with our EXL3 ours-K rows.
Each row is scaled to unit RMS (a per-output-channel scale, free in any epilogue) times a global factor searched on
{1, 1.15, 1.3} (env E3_FACTORS overrides). E3_SEED: the input rotation's sign seed (default 0);
E3_DAMP: LDLQ damping x mean diag(H) (default 0.01;
exllamav3's sigma_reg default is 0.025). E3_ROWSEED: which row sample (default 0) -- for the best-of-N seed test (seedtest.sh).
CODEBOOKS (comma list), each a 65536-entry table indexed by the 16-bit state, standardised to unit variance:
  mul1        exllamav3's mul1 (bytesum of state * 0x83DCD12D), in float
  mul1_i8     mul1 rounded to an int8 grid (+-127 = 3.5 sd), i.e. what an int8 WMMA would multiply
  gauss_i8    a Gaussian-quantile table on the int8 grid (state hash rank -> inverse normal CDF, rounded), the
              'designed for int8' alternative
  int4grid    mul1 rounded to 16 levels (int4 grid, +-7.5 = 2.6 sd): codes that could feed the iu4 path
"""
import ctypes, json, math, os, sys, time
import numpy as np
import torch
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3
torch.set_grad_enabled(False)
OUT = None
LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libviterbi.so"))
LIB.viterbi_tiles.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
LIB.viterbi_tiles_pat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
LIB.viterbi_tiles_tb.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
if os.environ.get("E3_F16") == "1": LIB.viterbi_set_f16(1)   # emulate the GPU search's fp16 costs
from fractions import Fraction


def pattern(K):
    """bits per step for K = a/b: b steps, a bits, spread evenly"""
    f = Fraction(K).limit_denominator(8); a, b = f.numerator, f.denominator
    return [((i + 1) * a) // b - (i * a) // b for i in range(b)]


def codebook(name):
    s = np.arange(65536, dtype=np.uint64)
    h = (s * 0x83DCD12D) & 0xFFFFFFFF
    v = sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    v = (v - v.mean()) / v.std()
    if name == "mul1": return v
    if name == "mul1_i8":
        st = 3.5 / 127; return np.clip(np.round(v / st), -127, 127) * st
    if name == "gauss_i8":
        r = np.argsort(np.argsort(h.astype(np.float64) + s.astype(np.float64) * 1e-12))   # rank of the hash
        nd = NormalDist(); g = np.array([nd.inv_cdf((k + 0.5) / 65536) for k in range(65536)])[r]
        st = 3.5 / 127; q = np.clip(np.round(g / st), -127, 127) * st
        return q / q.std()
    if name == "int4grid":
        st = 2.6 / 7.5; q = np.clip(np.round(v / st - 0.5) + 0.5, -7.5, 7.5) * st
        return q / q.std()
    raise SystemExit(f"unknown codebook {name}")


def vit(x, cb, K):
    """x [ntiles, 256] float32 -> quantized [ntiles, 256]"""
    x = np.ascontiguousarray(x, dtype=np.float32); out = np.empty_like(x)
    TB = int(os.environ.get("E3_TB", 0))     # >0: tail-biting search (EXL3's real constraint), TB wrap candidates
    if TB and float(K).is_integer():
        LIB.viterbi_tiles_tb(x.ctypes.data, out.ctypes.data, x.shape[0], cb.ctypes.data, int(K), TB)
        return out
    p = np.ascontiguousarray(pattern(K), dtype=np.int32)
    LIB.viterbi_tiles_pat(x.ctypes.data, out.ctypes.data, x.shape[0], cb.ctypes.data, p.ctypes.data, len(p))
    return out


def ldlq_trellis(W, H, cb, K, damp=0.01):
    """W [R, Kd] (R % 16 == 0, unit-RMS rows), H [Kd, Kd] -> Q"""
    W = W.clone(); R, Kd = W.shape
    H = H.clone(); dead = H.diagonal() == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for a in range(0, Kd, 16):
        blk = W[:, a:a + 16]
        tiles = blk.reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()
        q = torch.from_numpy(vit(tiles, cb, K).astype(np.float64)).reshape(R, 16)
        Q[:, a:a + 16] = q
        if a + 16 < Kd:
            E = (blk - q) @ torch.linalg.inv(U[a:a + 16, a:a + 16])
            W[:, a + 16:] -= E @ U[a:a + 16, a + 16:]
    return Q


def run(layer, tensor, cbs, bits, rows):
    names, hn = e3.ROUTE[tensor]
    W0 = e3.load_w(layer, names); H0 = e3.load_h(layer, hn)
    N, Kd = W0.shape
    g = torch.Generator().manual_seed(7 + layer + 100003 * int(os.environ.get("E3_ROWSEED", 0)))
    idx = torch.randperm(N, generator=g)[:rows].sort()[0]
    rk = e3.had_blocks(Kd, 1000 + layer + 7919 * int(os.environ.get("E3_SEED", 0)))
    Wr = rk(W0[idx]); Hr = rk(rk(H0).T).T
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    out = []
    for name in cbs:
        cb = np.ascontiguousarray(codebook(name), dtype=np.float32)
        for K in bits:
            t0 = time.time(); best = None; F = tuple(float(v) for v in os.environ.get("E3_FACTORS", "1.0,1.15,1.3").split(","))
            # the scale factors' rows are independent given H: one LDLQ pass over all of them stacked
            Qs = ldlq_trellis(torch.cat([Wr / (rms * f) for f in F], 0), Hr, cb, K, float(os.environ.get("E3_DAMP", 0.01)))
            allf = {}
            for i, f in enumerate(F):
                r = e3.rel(Wr, Qs[i * rows:(i + 1) * rows] * rms * f, Hr); allf[f] = r
                if best is None or r < best[0]: best = (r, f)
            o = {"layer": layer, "w": tensor, "method": "trellis_" + name, "seed": int(os.environ.get("E3_SEED", 0)),
                 "rowseed": int(os.environ.get("E3_ROWSEED", 0)), "damp": float(os.environ.get("E3_DAMP", 0.01)), "f16": os.environ.get("E3_F16") == "1", "tb": int(os.environ.get("E3_TB", 0)), "bits": K, "bpw": K + 16 / Kd,
                 "rel": best[0], "scale_f": best[1], **({"allf": allf} if os.environ.get("E3_ALLF") else {}), "rows": rows, "N": N, "K": Kd, "sec": round(time.time() - t0, 1)}
            print(json.dumps(o), flush=True); out.append(o)
            if OUT:
                with open(OUT, "a") as fh: fh.write(json.dumps(o) + "\n")
    return out


if __name__ == "__main__":
    layer, tensor = int(sys.argv[1]), sys.argv[2]
    cbs = sys.argv[3].split(","); bits = [float(Fraction(x)) for x in sys.argv[4].split(",")]
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    OUT = sys.argv[6] if len(sys.argv) > 6 else None
    run(layer, tensor, cbs, bits, rows)
