#!/usr/bin/env python3
"""settle91.py LAYER TENSOR SAMPLE OUTDIR -- ../w91.md item B6: damping normalization on w87's held-out two-sided proxy.
settle87.job unchanged, run with sigma_eff = sigma_cur x N(H_A) / mean diag(H_A) for N in (mean, med, trl):
  mean = mean diag(H_A) (today), med = median diag(H_A), trl = (tr H_A - lambda_max(H_A)) / (n - 1).
Levels with sigma_cur 0.005 (i8 K2.5, i8 K3, i4 K1 for an input-group tensor) go to OUTDIR/k1, i4 K0.5 (sigma_cur 0.05)
to OUTDIR/k05; settle87 writes one file per sigma_eff (LAYER-TENSOR-s{SAMPLE}-d{sigma}.jsonl). OUTDIR/LAYER-TENSOR-
s{SAMPLE}.norm.json maps each normalization to its sigma_eff (written first; the job is restartable as settle87's).
Env as settle87 (E3_CKPT, E3_HESS, SETTLE_CACHE, W77_SCALES, SETTLE_ORDER=rev, W77_HA/HB1/HB2, W87_HOUT)."""
import json, os, sys
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w87")]
import settle87
from settle87 import settle77, e3

layer, tensor, sample, outdir = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(outdir, exist_ok=True)
stem = os.path.join(outdir, f"{layer}-{tensor}-s{sample}")
nf = stem + ".norm.json"
if os.path.exists(nf): N = json.load(open(nf))
else:
    HA = settle77.load_h(os.environ["W77_HA"], layer, e3.ROUTE[tensor][1])
    d = HA.diagonal()
    lam = float(torch.linalg.eigvalsh(HA)[-1])
    N = {"mean": float(d.mean()), "med": float(d.median()), "trl": float((d.sum() - lam) / (len(d) - 1)),
         "lambda_max": lam, "argmax_diag": int(d.argmax()), "diag_max_share": float(d.max() / d.sum()), "n": len(d)}
    del HA
    json.dump(N, open(nf + ".part", "w"), indent=1); os.replace(nf + ".part", nf)
for sub, levels in (("k1", [("i8", "2.5"), ("i8", "3"), ("i4", "1")]), ("k05", [("i4", "0.5")])):
    sc = settle87.sigma_current(tensor, float(levels[0][1]))
    assert all(settle87.sigma_current(tensor, float(k)) == sc for _, k in levels)
    sig = {n: sc * N[n] / N["mean"] for n in ("mean", "med", "trl")}
    od = os.path.join(outdir, sub); os.makedirs(od, exist_ok=True)
    json.dump({"sigma_cur": sc, "sigma_eff": sig}, open(os.path.join(od, f"{layer}-{tensor}-s{sample}.sig.json"), "w"))
    settle87.SIG = list(sig.values()); settle87.LEVELS = levels
    settle87.job(layer, tensor, sample, os.path.join(outdir, sub))
open(stem + ".DONE", "w").close()
print("SETTLE91 DONE", layer, tensor, sample, json.dumps(N))
