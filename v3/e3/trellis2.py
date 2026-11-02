#!/usr/bin/env python3
"""trellis2.py LAYER TENSOR CODEBOOKS BITS [ROWS] [OUT.jsonl] -- CPU proxy tests of two ranked ideas
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    for name in cbs:
        cb = np.ascontiguousarray(codebook(name), dtype=np.float32)
        for K in bits:
            t0 = time.time()
            # the scale factor: chosen by the LDLQ pass over all factors stacked (as trellis.py), then CD on that one
            Qs = trellis.ldlq_trellis(torch.cat([Wr / (rms * f) for f in F], 0), Hr, cb, K, damp)
            rf = [e3.rel(Wr, Qs[i * rows:(i + 1) * rows] * rms * f, Hr) for i, f in enumerate(F)]
            f = F[int(np.argmin(rf))]
            def report(stage, Q, extra=None):
                o = {"layer": layer, "w": tensor, "method": f"{stage}_{name}", "bits": K, "rel": e3.rel(Wr, Q * rms * f, Hr),
                     "scale_f": f, "damp": damp, "seed": int(os.environ.get("E3_SEED", 0)), "rows": rows, "N": N, "K": Kd,
                     "sec": round(time.time() - t0, 1), **(extra or {})}
                print(json.dumps(o), flush=True)
                if out_path:
                    with open(out_path, "a") as fh: fh.write(json.dumps(o) + "\n")
            ldlq_cd(Wr / (rms * f), Hr, damp, cb, K, sweeps if os.environ.get("CD", "1") == "1" else 0, report)


if __name__ == "__main__":
    layer, tensor = int(sys.argv[1]), sys.argv[2]
    cbs = sys.argv[3].split(","); bits = [float(Fraction(x)) for x in sys.argv[4].split(",")]
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    run(layer, tensor, cbs, bits, rows, sys.argv[6] if len(sys.argv) > 6 else None)
