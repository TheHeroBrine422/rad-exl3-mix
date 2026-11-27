#!/usr/bin/env python3
"""ours_metrics.py LAYER EXPERT KIND OUT.jsonl -- w211's adopted baseline (Hadamard, per-expert H, damp 0.01, factors {1, 1.15, 1.3}, mad,
LDLQ, TB) re-run with K = W235_KS (default 1,2,3) on w211's rows, keeping the winning factor's reconstruction to report, in the ORIGINAL
(unrotated) domain, rel (full H; must equal w211's min over the three factors), rel_diag (diag H) and rel_mse (identity).
Same pipeline as w211/ds_check.py (setting 'base' restricted to F3); rows from the same seeded permutation (W211_SLICE, W235_ROWS_GU / _DN).
"""
import json, os, sys, time
sys.path.insert(0, os.environ.get("W235_W211DIR", "/data/q27/w235/code/w211"))
import numpy as np
import torch
import dscommon as D
import e3, trellis
torch.set_grad_enabled(False); torch.set_num_threads(1)
F3 = (1.0, 1.15, 1.3)


def main():
    layer, e, kind, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    Ks = [float(k) for k in os.environ.get("W235_KS", "1,2,3").split(",")]
    sl = int(os.environ.get("W211_SLICE", "0")); chunk = int(os.environ.get("W211_CHUNK", "512"))
    nrows = int(os.environ.get("W235_ROWS_GU" if kind == "gate_up" else "W235_ROWS_DN", "0"))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["K"], r["rows"], r["slice"]))
    Ks = [k for k in Ks if (k, nrows, sl) not in done]
    if not Ks: return
    t0 = time.time()
    W0 = D.load_expert_w(layer, e, kind); N, Kd = W0.shape
    H0, stats = D.load_expert_h(layer, e, kind); hdiag = H0.diagonal().clone()
    rk = e3.had_blocks(Kd, 1000 + layer)
    if nrows:
        g = torch.Generator().manual_seed(7 + layer + 100003 * e)
        idx = torch.randperm(N, generator=g)[sl * nrows:(sl + 1) * nrows].sort()[0]; assert len(idx) == nrows
        W0 = W0[idx]; N = nrows
    Wr = rk(W0)
    Hown = rk(rk(H0).T).T
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    Ws = Wr / rms
    Wr32 = Wr.float(); Hown32 = Hown.float()
    def quad(Dm, H32): return float(torch.einsum("ij,ij->", Dm @ H32, Dm).double())
    den_own = quad(Wr32, Hown32)
    den_diag = float((W0 ** 2 @ hdiag).sum()); den_mse = float((W0 ** 2).sum())
    den_full0 = float(torch.einsum("ij,jk,ik->", W0, H0, W0))        # unrotated full-H denominator (same evaluation as heph_check.py)
    pre = D.prep_ldlq(Hown, 0.01)
    # inverse of the rotation: Wr = rk(W0) is orthogonal, so W0 = rk^T(Wr); apply it via the dense 128-block matrix
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h /= 128 ** 0.5
    gg = torch.Generator().manual_seed(1000 + layer)
    sg = (torch.randint(0, 2, (Kd,), generator=gg) * 2 - 1).double()
    def unrot(y): return (y.reshape(*y.shape[:-1], Kd // 128, 128) @ h).reshape(y.shape) * sg
    assert float((unrot(Wr) - W0).abs().max()) < 1e-9
    del Hown; t_setup = time.time() - t0
    cbs = np.ascontiguousarray(trellis.codebook("mad"), dtype=np.float32)
    for K in Ks:
        tk = time.time()
        Q = torch.empty(len(F3), N, Kd, dtype=torch.float32)
        for r0 in range(0, N, chunk):
            r1 = min(r0 + chunk, N)
            Qc = D.ldlq_prepared(torch.cat([Ws[r0:r1] / f for f in F3], 0), pre, cbs, K)
            for i in range(len(F3)): Q[i, r0:r1] = Qc[i * (r1 - r0):(i + 1) * (r1 - r0)].float()
        allf = {}; best = None
        for i, f in enumerate(F3):
            Dm = Wr32 - Q[i] * (rms.float() * f); allf[str(f)] = quad(Dm, Hown32) / den_own
            if best is None or allf[str(f)] < best[0]: best = (allf[str(f)], i, f)
        i, f = best[1], best[2]
        What = unrot((Q[i].double() * (rms * f)))
        Dm = W0 - What
        rec = {"layer": layer, "e": e, "kind": kind, "K": K, "rows": nrows, "slice": sl, "R": N, "Kd": Kd, "factor": f,
               "rel": best[0], "rel_unrot": float(torch.einsum("ij,jk,ik->", Dm, H0, Dm)) / den_full0, "allf": allf, "rel_diag": float((Dm ** 2 @ hdiag).sum()) / den_diag, "rel_mse": float((Dm ** 2).sum()) / den_mse,
               "setup_sec": round(t_setup, 1), "sec": round(time.time() - tk, 1)}
        print(json.dumps(rec), flush=True)
        with open(out, "a") as fh: fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__": main()
