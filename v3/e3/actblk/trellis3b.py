#!/usr/bin/env python3
"""actblk/trellis3b.py -- v3/e3/trellis3.py plus E3_ORDER=blk128 (#103, actblk/README.md): whole 128-column blocks
(the Hadamard blocks) quantized in descending order of their block trace(H) (rotated H; the trace of a block is the
same before and after its rotation), sub-blocks inside a block in natural order. Same handling as hdiag (W and H
permuted, LDLQ in that order, codes permuted back, error in the original order); lines carry "moved" (share of
128-blocks not in place) and "mass_ratio" (largest / smallest block trace). The parent dir (v3/e3) must be on the
path (e3.py, trellis.py, libviterbi.so). Everything else is trellis3.py's, line for line:

trellis3.py LAYER TENSOR CODEBOOKS BITS [ROWS] [OUT.jsonl] -- trellis2.py plus two options for #95 (IQR-V4 ideas,
v3/e3/sub1/README.md "#95"). With both unset the output is trellis2.py's, line for line.

E3_ORDER   comma list of column orders, each run in turn on the same rows (unset: natural, and no "order" key):
    natural  stored order
    hdiag    95-01 act order: inside each 128-column Hadamard block, the 8 16-column sub-blocks are quantized in
             descending order of their H_in diagonal mass (rotated H, the metric LDLQ sees). W and H are permuted,
             LDLQ (+ CD) runs in that order, the codes are permuted back and the error is measured in the original
             column order. Lines carry "order" and, for hdiag, "moved" (share of 128-blocks whose order changes)
             and "mass_cv" (mean coefficient of variation of the 8 sub-block masses in a block).
E3_RS128=1 95-02: after the trellis (LDLQ stage), one bf16 scale per (row, 128-column block) on top of the row scale,
           closed form under the block's own H_in, s = (w_b H_bb q_b^T) / (q_b H_bb q_b^T), rounded to bf16; refit at
           every global factor, the best kept. Extra lines: method "rs128_<cb>" (that refit) and "rs128j_<cb>" (all
           of a row's block scales solved jointly under the full H, same factor; reported only), with "rel_base"
           (the same codes without the refit, best factor) and "extra_bpw" 0.125 (16 bits / 128 weights).

trellis2.py's description:
CPU proxy tests of two ranked ideas
(docs/ideas/2026-11-02/ranked.md) on real 27B tensors, on top of trellis.py's block-LDLQ + bitshift-trellis model:

C2  tile coordinate descent after LDLQ (env CD_SWEEPS, default 3). With the codes from LDLQ, sweep the 16-column
    blocks again; each block's unconstrained optimum given every other block is
        Q_a* = W_a + E_-a H_-a,a H_aa^-1      (E = W - Q, H the damped LDLQ metric),
    re-run the trellis on it, and keep a tile's new codes only if the exact change of tr(E H E^T),
        delta = 2 D . (E H)_a + D H_aa D^T   (D = old - new codes of that tile's rows),
    is negative. E H is kept up to date (one R x 16 x Kd product per block). The proxy error rel (undamped H, as
    trellis.py) is reported after LDLQ ("ldlq") and after each sweep ("cd1", "cd2", ...). Same rows, scale factor
    (the LDLQ pass's best), seed and codebook for both, so the comparison is paired.
D3  "Gaussianized" mul1 codebooks (CODEBOOKS names, besides trellis.py's mul1 / mul1_i8 / gauss_i8 / int4grid):
    gmul1_i8   mul1's bytesum mapped through its exact distribution (sum of 4 uniform bytes) to Gaussian quantiles,
               then the int8 grid -- the same codes, a value table of 1,021 entries
    cubicC_i8  v + C v^3 on the standardised bytesum (C as in "cubic0.04_i8"), restandardised, int8 grid -- 2 FMAs in
               the decode
Output: one JSON line per (codebook, K, stage) as trellis.py, method "ldlq_<cb>" / "cd<n>_<cb>". Env as trellis.py
(E3_SEED, E3_ROWSEED, E3_DAMP, E3_FACTORS, E3_CKPT, E3_HESS)."""
import json, math, os, sys, time
from fractions import Fraction
import numpy as np
import torch
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import e3, trellis
torch.set_grad_enabled(False)


def int8_grid(v):
    st = 3.5 / 127; q = np.clip(np.round(v / st), -127, 127) * st
    return q / q.std()


def codebook(name):
    if name.startswith(("gmul1", "cubic")):
        s = np.arange(65536, dtype=np.uint64)
        h = (s * 0x83DCD12D) & 0xFFFFFFFF
        bs = sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.int64)
        if name == "gmul1_i8":
            p = np.ones(256)
            for _ in range(3): p = np.convolve(p, np.ones(256))
            p /= p.sum(); cdf = np.cumsum(p) - p / 2              # mid-CDF of each bytesum value 0..1020
            nd = NormalDist(); g = np.array([nd.inv_cdf(min(max(c, 1e-9), 1 - 1e-9)) for c in cdf])
            v = g[bs]; return int8_grid((v - v.mean()) / v.std())
        if name.startswith("cubic") and name.endswith("_i8"):
            c = float(name[5:-3])
            v = bs.astype(np.float64); v = (v - v.mean()) / v.std(); v = v + c * v ** 3
            return int8_grid((v - v.mean()) / v.std())
    return trellis.codebook(name)


def ldlq_cd(W, H, Hd_damp, cb, K, sweeps, report):
    """trellis.ldlq_trellis, then `sweeps` tile coordinate-descent sweeps; report(stage, Q) after each"""
    Q = trellis.ldlq_trellis(W, H, cb, K, Hd_damp); report("ldlq", Q)
    if not sweeps: return
    R, Kd = W.shape
    Hd = H.clone(); dead = Hd.diagonal() == 0
    Hd[dead, dead] = 1; Wz = W.clone(); Wz[:, dead] = 0
    Hd.diagonal().add_(Hd_damp * Hd.diagonal().mean())
    E = Wz - Q; G = E @ Hd
    for sw in range(1, sweeps + 1):
        acc = tot = 0
        for a in range(0, Kd, 16):
            Haa = Hd[a:a + 16, a:a + 16]
            Ea = E[:, a:a + 16]
            T = Q[:, a:a + 16] + torch.linalg.solve(Haa, (G[:, a:a + 16]).T).T   # = W_a + E_-a H_-a,a H_aa^-1
            tiles = T.reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()
            qn = torch.from_numpy(trellis.vit(tiles, cb, K).astype(np.float64)).reshape(R, 16)
            D = Q[:, a:a + 16] - qn                                   # E_new - E_old for these columns
            dr = 2 * (D * G[:, a:a + 16]).sum(1) + ((D @ Haa) * D).sum(1)   # per row
            dt = dr.reshape(R // 16, 16).sum(1)                       # per tile (16 rows)
            keep = (dt < 0).repeat_interleave(16)
            tot += R // 16; acc += int((dt < 0).sum())
            if keep.any():
                Dk = D * keep[:, None]
                Q[:, a:a + 16] = torch.where(keep[:, None], qn, Q[:, a:a + 16])
                E[:, a:a + 16] += Dk; G += Dk @ Hd[a:a + 16, :]
        report(f"cd{sw}", Q, {"accepted": acc / tot})


def hdiag_perm(H):
    """column permutation: inside each 128-block, 16-column sub-blocks by descending diag(H) mass; + stats"""
    Kd = H.shape[0]; nb = Kd // 128
    m = H.diagonal().reshape(nb, 8, 16).sum(-1)
    o = torch.argsort(-m, dim=1, stable=True)
    perm = ((torch.arange(nb)[:, None] * 8 + o)[:, :, None] * 16 + torch.arange(16)).reshape(-1)
    moved = float((o != torch.arange(8)).any(1).double().mean())
    cv = float((m.std(1) / m.mean(1).clamp(min=1e-300)).mean())
    return perm, {"moved": moved, "mass_cv": cv}


def blk128_perm(H):
    """column permutation: whole 128-blocks by descending block trace(H), stable; + stats"""
    Kd = H.shape[0]; nb = Kd // 128
    m = H.diagonal().reshape(nb, 128).sum(-1)
    o = torch.argsort(-m, stable=True)
    perm = (o[:, None] * 128 + torch.arange(128)).reshape(-1)
    return perm, {"moved": float((o != torch.arange(nb)).double().mean()),
                  "mass_ratio": float(m.max() / m.min().clamp(min=1e-300))}


def rs128(W, Q, H):
    """per (row, 128-block) bf16 scale, closed form under the block's H_bb -> refit Q"""
    R, Kd = W.shape; nb = Kd // 128
    Hb = torch.stack([H[i * 128:(i + 1) * 128, i * 128:(i + 1) * 128] for i in range(nb)])
    Qb = Q.reshape(R, nb, 128); QH = torch.einsum("rbi,bij->rbj", Qb, Hb)
    num = (QH * W.reshape(R, nb, 128)).sum(-1); den = (QH * Qb).sum(-1)
    s = torch.where(den > 0, num / den.clamp(min=1e-300), torch.ones_like(den)).to(torch.bfloat16).double()
    return (Qb * s[..., None]).reshape(R, Kd), s


def rs128j(W, Q, H, chunk=8):
    """the same scales solved jointly per row under the full H: G s = c, G_bc = q_b H_bc q_c^T, c_b = q_b H_b: w^T"""
    R, Kd = W.shape; nb = Kd // 128; Hv = H.reshape(nb, 128, Kd); out = []; ss = []
    for r0 in range(0, R, chunk):
        Qb = Q[r0:r0 + chunk].reshape(-1, nb, 128)
        M = torch.einsum("rbi,bij->rbj", Qb, Hv)
        G = torch.einsum("rbcj,rcj->rbc", M.reshape(-1, nb, nb, 128), Qb)
        c = torch.einsum("rbj,rj->rb", M, W[r0:r0 + chunk])
        G = G + 1e-9 * G.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(nb, dtype=G.dtype)
        s = torch.linalg.solve(G, c[..., None])[..., 0].to(torch.bfloat16).double()
        out.append((Qb * s[..., None]).reshape(-1, Kd)); ss.append(s)
    return torch.cat(out), torch.cat(ss)


def run(layer, tensor, cbs, bits, rows, out_path):
    names, hn = e3.ROUTE[tensor]
    W0 = e3.load_w(layer, names); H0 = e3.load_h(layer, hn)
    N, Kd = W0.shape
    g = torch.Generator().manual_seed(7 + layer + 100003 * int(os.environ.get("E3_ROWSEED", 0)))
    idx = torch.randperm(N, generator=g)[:rows].sort()[0]
    rk = e3.had_blocks(Kd, 1000 + layer + 7919 * int(os.environ.get("E3_SEED", 0)))
    Wr = rk(W0[idx]); Hr = rk(rk(H0).T).T
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    damp = float(os.environ.get("E3_DAMP", 0.01)); sweeps = int(os.environ.get("CD_SWEEPS", 3))
    F = tuple(float(v) for v in os.environ.get("E3_FACTORS", "1.0,1.15,1.3").split(","))
    orders = os.environ.get("E3_ORDER"); orders = orders.split(",") if orders else [None]
    rs = os.environ.get("E3_RS128") == "1"
    for name in cbs:
        cb = np.ascontiguousarray(codebook(name), dtype=np.float32)
        for K in bits:
          for order in orders:
            t0 = time.time()
            if order in (None, "natural"): perm, inv, ox = None, None, {}
            elif order == "hdiag": perm, ox = hdiag_perm(Hr); inv = torch.argsort(perm)
            elif order == "blk128": perm, ox = blk128_perm(Hr); inv = torch.argsort(perm)
            else: raise SystemExit(f"unknown E3_ORDER {order}")
            if order is not None: ox = {"order": order, **ox}
            Wq, Hq = (Wr, Hr) if perm is None else (Wr[:, perm], Hr[perm][:, perm])   # quantization order
            back = (lambda Q: Q) if perm is None else (lambda Q: Q[:, inv])            # codes -> original order
            # the scale factor: chosen by the LDLQ pass over all factors stacked (as trellis.py), then CD on that one
            Qs = back(trellis.ldlq_trellis(torch.cat([Wq / (rms * f) for f in F], 0), Hq, cb, K, damp))
            rf = [e3.rel(Wr, Qs[i * rows:(i + 1) * rows] * rms * f, Hr) for i, f in enumerate(F)]
            f = F[int(np.argmin(rf))]
            def emit(o):
                print(json.dumps(o), flush=True)
                if out_path:
                    with open(out_path, "a") as fh: fh.write(json.dumps(o) + "\n")
            def report(stage, Q, extra=None):
                o = {"layer": layer, "w": tensor, "method": f"{stage}_{name}", "bits": K, "rel": e3.rel(Wr, back(Q) * rms * f, Hr),
                     "scale_f": f, "damp": damp, "seed": int(os.environ.get("E3_SEED", 0)), "rows": rows, "N": N, "K": Kd,
                     "sec": round(time.time() - t0, 1), **(extra or {}), **ox}
                emit(o)
            ldlq_cd(Wq / (rms * f), Hq, damp, cb, K, sweeps if os.environ.get("CD", "1") == "1" else 0, report)
            if rs:
                t1 = time.time(); fits = []
                for i, fi in enumerate(F):
                    Qf = Qs[i * rows:(i + 1) * rows] * rms * fi
                    Qr, s = rs128(Wr, Qf, Hr); fits.append((e3.rel(Wr, Qr, Hr), fi, Qf, s))
                r, fr, Qf, s = min(fits, key=lambda t: t[0])
                base = {"layer": layer, "w": tensor, "bits": K, "extra_bpw": 16 / 128, "rel_base": min(rf),
                        "scale_f_base": f, "damp": damp, "seed": int(os.environ.get("E3_SEED", 0)), "rows": rows, "N": N,
                        "K": Kd, **ox}
                d = (s - 1).abs().flatten()
                emit({**base, "method": f"rs128_{name}", "rel": r, "scale_f": fr, "s_dev_med": float(d.median()),
                      "s_dev_p90": float(d.quantile(0.9)), "sec": round(time.time() - t1, 1)})
                t1 = time.time(); Qj, sj = rs128j(Wr, Qf, Hr); d = (sj - 1).abs().flatten()
                emit({**base, "method": f"rs128j_{name}", "rel": e3.rel(Wr, Qj, Hr), "scale_f": fr,
                      "s_dev_med": float(d.median()), "s_dev_p90": float(d.quantile(0.9)), "sec": round(time.time() - t1, 1)})


if __name__ == "__main__":
    layer, tensor = int(sys.argv[1]), sys.argv[2]
    cbs = sys.argv[3].split(","); bits = [float(Fraction(x)) for x in sys.argv[4].split(",")]
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    run(layer, tensor, cbs, bits, rows, sys.argv[6] if len(sys.argv) > 6 else None)
