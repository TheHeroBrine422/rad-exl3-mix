#!/usr/bin/env python3
"""tensor_check.py LAYER TENSOR BITS [ROWS] [OUT.jsonl] -- CPTC vs EXL3 codebooks on ONE 27B tensor
(C3 of the CPTC workstream): the E3 hproxy relative Hessian error (v3/e3/e3.py's metric) of the
tail-biting trellis quantization (v3/e3/trellis.py's LDLQ machinery, E3_TB=16) with:

  - cptcK:      CPTC (arXiv 2611.22019) with a PER-TENSOR table fit on this tensor's rotated
                unit-RMS values (A.2 protocol; retargeting = the 64 table values only).
  - mul1, mul1_i8, gauss_i8, int4grid: the global EXL3 codebooks (trellis.py codebook()),
                the same trellis, same LDLQ -- isolates the codebook/format.

Quantizer-side proxy only (no inference cost; that is C4). The (layer, tensor) pair must exist in
the checkpoint (hproxy ROUTE names); rows are independent given H so ROWS (default 256) random
rows are unbiased, as in trellis.py.

Run per (layer, tensor) as one job; aggregate the OUT.jsonl lines afterwards.
  OMP_NUM_THREADS=4 E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib \
    python3 tensor_check.py 3 ffn_down 2,3,4 256 out.jsonl
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.environ.get("CPTC_E3DIR", os.path.join(HERE, "..", "v3", "e3"))
sys.path.insert(0, E3)
os.environ.setdefault("E3_TB", "16")   # tail-biting search (EXL3's real constraint), 16 wrap candidates
import e3, trellis
sys.path.insert(0, HERE)
import cptc

F = tuple(float(v) for v in os.environ.get("E3_FACTORS", "1.0,1.15,1.3").split(","))
BASELINES = os.environ.get("E3_C3_BASE", "mul1,mul1_i8,gauss_i8,int4grid").split(",")   # 11-20: use mul1,mad


def table_init_emp(vals, a0, a1, K):
    """Eq (2) init from the EMPIRICAL target CDF (tensor values): cell i owns [G(i-1), G(i)) of the
    index CDF; Tfull[i] = conditional mean of vals over that F interval."""
    s = np.arange(65536, dtype=np.uint32)
    h = np.bincount(cptc.cell_index(a0, a1, s, 0), minlength=127)
    G = np.cumsum(h) / h.sum()
    v = np.sort(np.asarray(vals, dtype=np.float64))
    n = v.size
    Fq = (np.arange(n) + 0.5) / n  # empirical CDF at each sample
    u = np.concatenate([[0.0], G[:-1]])
    uu = np.interp(u, Fq, v, left=v[0], right=v[-1])     # quantiles of the empirical CDF
    vv = np.interp(G, Fq, v, left=v[0], right=v[-1])
    Tfull = np.zeros(127)
    for i in range(127):
        m = (v >= uu[i]) & (v < vv[i])
        Tfull[i] = v[m].mean() if m.any() else (uu[i] + vv[i]) / 2
    nn = 16 - K + 1
    Tfull *= (1 + 2 / nn) * (1 - 4 ** -K)
    Tlower = np.zeros(64)
    Tlower[:63] = (Tfull[:63] - Tfull[126:63:-1]) / 2
    return Tlower


def fit_table(vals, K, rounds=50, snap_rounds=16, nblk_fit=1024, seed=1234):
    """A.2 protocol on a real tensor target (fresh subsamples per round, damping 0.5 as in cptc.py)."""
    a0, a1 = cptc.COEF[K]
    assert cptc.depth(a0, a1) == 16
    rng = np.random.default_rng(seed)
    T = table_init_emp(vals, a0, a1, K)
    best = None
    xh_idx = rng.choice(vals.size, 262144, replace=False)
    xh = vals[xh_idx].reshape(-1, 256).astype(np.float32)
    for r in range(rounds):
        damped = r >= 30
        xi = rng.choice(vals.size, nblk_fit * 256, replace=False)
        xf = vals[xi].reshape(-1, 256).astype(np.float32)
        Tnew, _, _ = cptc.lloyd_round(T, a0, a1, K, xf, ntry=1)
        T = 0.5 * T + 0.5 * Tnew if damped else Tnew
        qh, _ = cptc.encode_rev(xh, cptc.decode_table(a0, a1, T).astype(np.float32), K, 1)
        sc = -np.log2(np.sqrt(np.mean((xh - qh) ** 2)))
        if best is None or sc > best[0]:
            best = (sc, T.copy())
    T = best[1]
    grid = cptc.e2m5_pos()
    for r in range(snap_rounds):
        xi = rng.choice(vals.size, nblk_fit * 256, replace=False)
        xf = vals[xi].reshape(-1, 256).astype(np.float32)
        T, _, _ = cptc.lloyd_round(T, a0, a1, K, xf, ntry=1)
        T = cptc.snap_e2m5(T, grid)
    return T


def refit_table(T, K, Wn, Hr):
    """One per-tensor refit (factor 1): code the tensor with table T through LDLQ, then set each entry to the mirrored
    mean of the LDLQ targets coded to it (entries matched by magnitude; plain mean, the block D weights ignored), and
    snap to the storage grid (CPTC_GRID). Entries never used keep their value; the zero entry stays 0."""
    cb = np.ascontiguousarray(cptc.decode_table(*cptc.COEF[K], T), dtype=np.float32)
    col = []
    trellis.ldlq_trellis(Wn, Hr, cb, K, float(os.environ.get("E3_DAMP", "0.01")), collect=col)
    t = np.concatenate([c[0].ravel() for c in col]); q = np.concatenate([c[1].ravel() for c in col]).astype(np.float32)
    mags = np.unique(np.abs(T.astype(np.float32)))
    out = T.copy()
    aq = np.abs(q)
    j = np.searchsorted(mags, aq)
    j = np.clip(j, 0, len(mags) - 1)
    ok = np.isclose(mags[j], aq, rtol=0, atol=1e-6)
    sg = np.sign(q)
    for m_i, m in enumerate(mags):
        if m == 0:
            continue
        sel = ok & (j == m_i)
        if sel.sum() < 4:
            continue
        newm = float(np.mean(t[sel] * sg[sel]))
        out[np.isclose(np.abs(T), m, rtol=0, atol=1e-6)] = np.sign(T[np.isclose(np.abs(T), m, rtol=0, atol=1e-6)]) * max(newm, 0)
    return cptc.snap_e2m5(out, cptc.table_grid())


def main():
    layer, tensor = int(sys.argv[1]), sys.argv[2]
    bits = [int(x) for x in sys.argv[3].split(",")]
    rows = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    OUT = sys.argv[5] if len(sys.argv) > 5 else None

    names, hn = e3.ROUTE[tensor]
    t0 = time.time()
    W0 = e3.load_w(layer, names)
    try:
        H0 = e3.load_h(layer, hn)
    except FileNotFoundError:
        print(json.dumps({"layer": layer, "w": tensor, "skip": "no hessian"}), flush=True)
        return
    N, Kd = W0.shape
    g = torch.Generator().manual_seed(7 + layer + 100003)
    idx = torch.randperm(N, generator=g)[:rows].sort()[0]
    rk = e3.had_blocks(Kd, 1000 + layer)
    Wr = rk(W0[idx])
    Hr = rk(rk(H0).T).T
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    Ws = (Wr / rms).double()                       # unit-RMS rotated rows (factor 1)
    flat = Ws.numpy().ravel()

    res2 = {"layer": layer, "w": tensor, "rows": rows, "N": N, "K": Kd, "per_K": {}}
    t_fit = 0.0
    for K in bits:
        cbs = {n: np.ascontiguousarray(trellis.codebook(n), dtype=np.float32) for n in BASELINES}
        tf = time.time()
        tdir = os.environ.get("CPTC_TABLES")   # global Gaussian tables (the paper's deployed tables), else per-tensor fit
        if os.environ.get("E3_C3_ARMS", "all") == "cptc": cbs = {}
        if os.environ.get("E3_C3_ARMS", "all") != "base":
            T = np.load(os.path.join(tdir, f"table-K{K}.npy")) if tdir else fit_table(flat, K)
            cbs[f"cptc{K}"] = np.ascontiguousarray(cptc.decode_table(*cptc.COEF[K], T), dtype=np.float32)
            for it in range(int(os.environ.get("CPTC_REFIT", "0"))):   # per-tensor closed-form refit on the LDLQ targets
                T = refit_table(T, K, Wr / rms, Hr)
                cbs[f"cptc{K}r{it + 1}"] = np.ascontiguousarray(cptc.decode_table(*cptc.COEF[K], T), dtype=np.float32)
        for spec in filter(None, os.environ.get("CB_EXTRA", "").split(",")):   # name=DIR: DIR/<name>-K<K>.npy (65536 codebook)
            nm, d = spec.split("=", 1); f = os.path.join(d, f"{nm.replace(':', '_')}-K{K}.npy")
            if os.path.exists(f): cbs[nm.split(":")[0]] = np.ascontiguousarray(np.load(f), dtype=np.float32)
        t_fit += time.time() - tf
        row = {}
        for n, cb in cbs.items():
            Qs = trellis.ldlq_trellis(torch.cat([Wr / (rms * f) for f in F], 0), Hr, cb, K,
                                      float(os.environ.get("E3_DAMP", "0.01")))
            best = None
            for i, f in enumerate(F):
                r = e3.rel(Wr, Qs[i * rows:(i + 1) * rows] * rms * f, Hr)
                if best is None or r < best:
                    best = r
            row[n] = round(float(best), 9)
        res2["per_K"][str(K)] = row
    res2["fit_sec"] = round(t_fit, 1)
    res2["total_sec"] = round(time.time() - t0, 1)
    print(json.dumps(res2), flush=True)
    if OUT:
        with open(OUT, "a") as fh:
            fh.write(json.dumps(res2) + "\n")


if __name__ == "__main__":
    main()
