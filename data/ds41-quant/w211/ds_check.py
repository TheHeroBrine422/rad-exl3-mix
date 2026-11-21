#!/usr/bin/env python3
"""ds_check.py LAYER EXPERT KIND SETTING OUT.jsonl -- one (matrix, setting) job of the w211 grid, K = W211_KS (default 1,2,3,4).

KIND: gate_up (w1;w3 stacked, [4608,5120], H = expert h13) | down (w2, [5120,2304], H = h2).
SETTING (name -> damping, scale factors tried, Hessian used for the quantizer):
  base      damp 0.01, factors 0.8 0.9 1.0 1.15 1.3 1.5, own H   (baseline = best of {1, 1.15, 1.3}, wide grid = best of all six)
  d0.001 d0.005 d0.025 d0.05 d0.1   damp as named, factors {1, 1.15, 1.3}, own H
  pooled    damp 0.01, factors {1, 1.15, 1.3}, H = pooled layer ffn_in Hessian        (gate_up only)
  mix0.1 mix1   own H + lambda * pooled (pooled rescaled to the own H's mean diagonal) (gate_up only)
Everything else is the 27B pipeline (cptc/tensor_check.py): block-Hadamard rotation (seed 1000 + layer), unit-RMS rows, global factor
per matrix, LDLQ with relative damping, mad codebook, K integer, tail-biting TB=1 (E3_V2=3), all rows (W211_ROWS=0) unless a
seeded row subsample is requested (the same rows for every setting). Metric: H-weighted relative error under the EXPERT's own H
(rel_own; in-sample for the own-H settings) and, gate_up only, under the pooled H (rel_pool). Restartable: (matrix, setting, K) lines
already in OUT are skipped. One extra line {"meta": ...} per matrix from the base job (kurtosis of the rotated rows, kept rows).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import dscommon as D
import e3, trellis
torch.set_num_threads(1)

F3 = (1.0, 1.15, 1.3)
SETTINGS = {
    "base": (0.01, (0.8, 0.9, 1.0, 1.15, 1.3, 1.5), "own"),
    "d0.001": (0.001, F3, "own"), "d0.005": (0.005, F3, "own"), "d0.025": (0.025, F3, "own"),
    "d0.05": (0.05, F3, "own"), "d0.1": (0.1, F3, "own"),
    "pooled": (0.01, F3, "pooled"), "mix0.1": (0.01, F3, "mix0.1"), "mix1": (0.01, F3, "mix1"),
}


def sym_rot(rk, H):
    return rk(rk(H).T).T


def excess_kurtosis(x):
    x = x.reshape(-1).double(); m2 = (x * x).mean()
    return float((x ** 4).mean() / m2 ** 2 - 3)


def main():
    layer, e, kind, setting, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    Ks = [int(k) for k in os.environ.get("W211_KS", "1,2,3,4").split(",")]
    nrows = int(os.environ.get("W211_ROWS", "0"))      # 0 = all rows, else rows per slice
    sl = int(os.environ.get("W211_SLICE", "0"))        # disjoint row slices: rows perm[sl*nrows:(sl+1)*nrows] of one seeded permutation
    chunk = int(os.environ.get("W211_CHUNK", "512"))
    damp, Fs, hsel = SETTINGS[setting]
    if kind == "down" and hsel != "own":
        sys.exit("pooled / mix settings are gate_up only")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            if r.get("layer") == layer and r.get("e") == e and r.get("kind") == kind and r.get("setting") == setting and r.get("slice", 0) == sl and "K" in r:
                done.add(r["K"])
    Ks = [k for k in Ks if k not in done]
    base_meta_needed = setting == "base" and sl == 0 and not (os.path.exists(out) and any('"meta"' in l and f'"e": {e},' in l and f'"layer": {layer},' in l and f'"kind": "{kind}"' in l for l in open(out)))
    if not Ks and not base_meta_needed:
        return
    t0 = time.time()
    W0 = D.load_expert_w(layer, e, kind); N, Kd = W0.shape
    H0, stats = D.load_expert_h(layer, e, kind)
    Hp0 = D.load_pooled_h(layer) if kind == "gate_up" and (hsel != "own" or True) else None
    rk = e3.had_blocks(Kd, 1000 + layer)
    Wr = rk(W0)
    W0full, Wrfull = W0, Wr                              # meta statistics use every row
    if nrows:
        g = torch.Generator().manual_seed(7 + layer + 100003 * e)
        idx = torch.randperm(N, generator=g)[sl * nrows:(sl + 1) * nrows].sort()[0]; assert len(idx) == nrows
        W0 = W0[idx]; Wr = Wr[idx]; N = nrows
    Hown = sym_rot(rk, H0)
    Hpool = None
    if Hp0 is not None:
        Hp0 = Hp0 * (H0.diagonal().mean() / Hp0.diagonal().mean())     # same mean diagonal as the expert's H
        Hpool = sym_rot(rk, Hp0)
    del H0, Hp0
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    Ws = Wr / rms
    if base_meta_needed:
        Wsf = Wrfull / Wrfull.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        W0n = W0full / W0full.pow(2).mean(1, keepdim=True).sqrt()
        gen = torch.Generator().manual_seed(1)
        gs = torch.randn(Wsf.shape, generator=gen, dtype=torch.float64)
        sub = Wsf.reshape(-1)[torch.randperm(Wsf.numel(), generator=gen)[:2_000_000]]
        srt = torch.sort(sub)[0]; nd = torch.distributions.Normal(0., 1.)
        ks = float((torch.arange(1, srt.numel() + 1) / srt.numel() - nd.cdf(srt)).abs().max())
        meta = {"layer": layer, "e": e, "kind": kind, "meta": {
            "stats": stats, "N": int(W0full.shape[0]), "Kd": Kd, "kurt_rot": excess_kurtosis(Wsf), "kurt_unrot": excess_kurtosis(W0n),
            "kurt_rot_row_mean": float(((Wsf ** 4).mean(1) / (Wsf ** 2).mean(1) ** 2 - 3).mean()),
            "kurt_gauss_ref": excess_kurtosis(gs), "ks_rot": ks, "ks_gauss_ref_n": 2_000_000,
            "frac_zero_unrot": float((W0full == 0).double().mean()),
            "hdiag_mean": float(Hown.diagonal().mean()), "hdiag_dead": int((Hown.diagonal() == 0).sum()),
            "h_eff_rank_ratio": float(Hown.diagonal().sum() ** 2 / (Hown ** 2).sum())}}
        with open(out, "a") as fh: fh.write(json.dumps(meta) + "\n")
    Hq = {"own": Hown, "pooled": Hpool, "mix0.1": None if Hpool is None else Hown + 0.1 * Hpool,
          "mix1": None if Hpool is None else Hown + 1.0 * Hpool}[hsel]
    # evaluation denominators (fp32 matmuls: all terms PSD-positive, error ~1e-6 relative)
    Wr32 = Wr.float(); Hown32 = Hown.float(); Hpool32 = Hpool.float() if Hpool is not None else None
    def quad(Dm, H32): return float(torch.einsum("ij,ij->", Dm @ H32, Dm).double())
    den_own = quad(Wr32, Hown32); den_pool = quad(Wr32, Hpool32) if Hpool32 is not None else None
    pre = D.prep_ldlq(Hq, damp)
    del Hq, Hown, Hpool, W0full, Wrfull; import gc; gc.collect()      # keep the steady-state RSS ~1 GB (30 jobs share the box)
    t_setup = time.time() - t0
    cbs = np.ascontiguousarray(trellis.codebook("mad"), dtype=np.float32)
    nf = len(Fs)
    for K in Ks:
        tk = time.time()
        Q = torch.empty(nf, N, Kd, dtype=torch.float32)
        for r0 in range(0, N, chunk):
            r1 = min(r0 + chunk, N)
            Wc = torch.cat([Ws[r0:r1] / f for f in Fs], 0)
            Qc = D.ldlq_prepared(Wc, pre, cbs, K)
            for i in range(nf):
                Q[i, r0:r1] = Qc[i * (r1 - r0):(i + 1) * (r1 - r0)].float()
            del Wc, Qc
        allf = {}
        for i, f in enumerate(Fs):
            Dm = Wr32 - Q[i] * (rms.float() * f)
            allf[str(f)] = {"own": quad(Dm, Hown32) / den_own}
            if Hpool32 is not None: allf[str(f)]["pool"] = quad(Dm, Hpool32) / den_pool
        rec = {"layer": layer, "e": e, "kind": kind, "setting": setting, "slice": sl, "K": K, "damp": damp, "rows": N, "Kd": Kd,
               "hq": hsel, "allf": allf, "setup_sec": round(t_setup, 1), "sec": round(time.time() - tk, 1)}
        print(json.dumps(rec), flush=True)
        with open(out, "a") as fh: fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
