#!/usr/bin/env python3
"""trellis_chain.py -- #102 chained tail-biting on the CPU proxy (README.md here). Wraps chain/viterbi.c's
viterbi_chain_tb: the row tiles of one 16-column block (LDLQ quantizes them together) are cut into chains of n
consecutive tiles, each chain one tail-biting ring of 256 n positions (tile r starts from tile r-1's last word).

  trellis_chain.py gauss NTILES KS NS [OUT.jsonl]
      i.i.d. Gaussian tiles (seed 7, #57's check), mul1: free start, viterbi_tiles_tb (v3/e3, NTRY 1 and 16), and the
      chain at each n of NS (NTRY 1; n = 1 must equal viterbi_tiles_tb exactly at integer K). One line per (K, arm).
  trellis_chain.py tensor LAYER TENSOR CODEBOOK KS NS ROWS [OUT.jsonl]
      #57's proxy: trellis.py's block LDLQ on ROWS sampled rows (rotation seed 0, damp 0.01, factors 1/1.15/1.3
      stacked), same rows for every arm; arm n = 0 is the free start, n >= 1 a chain of n tiles (NTRY 1). The chains
      never cross the stacked factors (ROWS / 16 must be a multiple of n). One line per (K, n): rel (best factor).
KS may be fractional (2.5 = steps of [2, 3], trellis.pattern). Env E3_CKPT, E3_HESS, E3_DAMP, E3_FACTORS, CHAIN_NTRY."""
import ctypes, json, os, sys, time
from fractions import Fraction
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import e3, trellis
torch.set_grad_enabled(False)
LIB = ctypes.CDLL(os.path.join(HERE, "libviterbi.so"))
LIB.viterbi_chain_tb.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2
LIB.viterbi_tiles_tb.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
LIB.viterbi_tiles_pat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def vit_free(x, cb, K):
    x = np.ascontiguousarray(x, dtype=np.float32); out = np.empty_like(x)
    p = np.ascontiguousarray(trellis.pattern(K), dtype=np.int32)
    LIB.viterbi_tiles_pat(x.ctypes.data, out.ctypes.data, x.shape[0], cb.ctypes.data, p.ctypes.data, len(p))
    return out


def vit_chain(x, cb, K, n, ntry=1):
    """x [ntiles, 256], consecutive groups of n tiles chained -> quantized [ntiles, 256]"""
    nt = x.shape[0]; assert nt % n == 0, (nt, n)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(nt // n, 256 * n); out = np.empty_like(x)
    p = np.ascontiguousarray(trellis.pattern(K), dtype=np.int32); assert p.min() >= 1
    LIB.viterbi_chain_tb(x.ctypes.data, out.ctypes.data, None, nt // n, 256 * n, cb.ctypes.data, p.ctypes.data, len(p), ntry)
    return out.reshape(nt, 256)


def vit(x, cb, K, n, ntry=1):
    return vit_free(x, cb, K) if n == 0 else vit_chain(x, cb, K, n, ntry)


def emit(o, out):
    print(json.dumps(o), flush=True)
    if out:
        with open(out, "a") as fh: fh.write(json.dumps(o) + "\n")


def gauss(ntiles, ks, ns, out):
    x = np.random.default_rng(7).standard_normal((ntiles, 256)).astype(np.float32)
    cb = np.ascontiguousarray(trellis.codebook("mul1"), dtype=np.float32)
    e = lambda q: float(((q - x) ** 2).mean())
    for K in ks:
        base = {"mode": "gauss", "tiles": ntiles, "bits": K}
        t0 = time.time(); emit({**base, "arm": "free", "mse": e(vit_free(x, cb, K)), "sec": round(time.time() - t0, 1)}, out)
        if float(K).is_integer():
            for nt in (1, 16):
                t0 = time.time(); q = np.empty_like(x)
                LIB.viterbi_tiles_tb(x.ctypes.data, q.ctypes.data, ntiles, cb.ctypes.data, int(K), nt)
                ref = q if nt == 1 else ref
                emit({**base, "arm": f"tiles_tb{nt}", "mse": e(q), "sec": round(time.time() - t0, 1)}, out)
        for n in ns:
            if ntiles % n: continue
            t0 = time.time(); q = vit_chain(x, cb, K, n)
            o = {**base, "arm": f"chain{n}", "n": n, "mse": e(q), "sec": round(time.time() - t0, 1)}
            if n == 1 and float(K).is_integer(): o["identical_to_tiles_tb1"] = bool(np.array_equal(q, ref))
            emit(o, out)


def ldlq(W, U, dead, cb, K, n):
    """trellis.ldlq_trellis with U = chol(H_damped^-1) precomputed (shared by every arm) and the chain search"""
    W = W.clone(); R, Kd = W.shape; W[:, dead] = 0
    Q = torch.zeros_like(W)
    for a in range(0, Kd, 16):
        blk = W[:, a:a + 16]
        tiles = blk.reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()
        q = torch.from_numpy(vit(tiles, cb, K, n, int(os.environ.get("CHAIN_NTRY", 1))).astype(np.float64)).reshape(R, 16)
        Q[:, a:a + 16] = q
        if a + 16 < Kd:
            E = (blk - q) @ torch.linalg.inv(U[a:a + 16, a:a + 16])
            W[:, a + 16:] -= E @ U[a:a + 16, a + 16:]
    return Q


def tensor(layer, w, cbname, ks, ns, rows, out):
    names, hn = e3.ROUTE[w]
    W0 = e3.load_w(layer, names); H0 = e3.load_h(layer, hn)
    N, Kd = W0.shape
    g = torch.Generator().manual_seed(7 + layer + 100003 * int(os.environ.get("E3_ROWSEED", 0)))
    idx = torch.randperm(N, generator=g)[:rows].sort()[0]
    rk = e3.had_blocks(Kd, 1000 + layer + 7919 * int(os.environ.get("E3_SEED", 0)))
    Wr = rk(W0[idx]); del W0
    Hr = rk(rk(H0).T).T.contiguous(); del H0
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    damp = float(os.environ.get("E3_DAMP", 0.01))
    F = tuple(float(v) for v in os.environ.get("E3_FACTORS", "1.0,1.15,1.3").split(","))
    t0 = time.time()
    H = Hr.clone(); dead = H.diagonal() == 0
    H[dead, dead] = 1; H.diagonal().add_(damp * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True); del H
    tprep = time.time() - t0
    cb = np.ascontiguousarray(trellis.codebook(cbname), dtype=np.float32)
    Ws = torch.cat([Wr / (rms * f) for f in F], 0)
    for K in ks:
        for n in ns:
            assert n == 0 or (rows // 16) % n == 0
            t0 = time.time(); Qs = ldlq(Ws, U, dead, cb, K, n)
            allf = {f: e3.rel(Wr, Qs[i * rows:(i + 1) * rows] * rms * f, Hr) for i, f in enumerate(F)}
            f = min(allf, key=allf.get)
            emit({"mode": "tensor", "layer": layer, "w": w, "cb": cbname, "bits": K, "n": n, "ntry": int(os.environ.get("CHAIN_NTRY", 1)),
                  "rel": allf[f], "scale_f": f, "allf": allf, "damp": damp, "rows": rows, "N": N, "K": Kd,
                  "sec": round(time.time() - t0, 1), "prep_sec": round(tprep, 1)}, out)


if __name__ == "__main__":
    a = sys.argv[1:]
    ks = lambda s: [float(Fraction(v)) for v in s.split(",")]
    if a[0] == "gauss":
        gauss(int(a[1]), ks(a[2]), [int(v) for v in a[3].split(",")], a[4] if len(a) > 4 else None)
    elif a[0] == "tensor":
        tensor(int(a[1]), a[2], a[3], ks(a[4]), [int(v) for v in a[5].split(",")], int(a[6]), a[7] if len(a) > 7 else None)
    else:
        raise SystemExit(__doc__)
