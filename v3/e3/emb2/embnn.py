#!/usr/bin/env python3
"""embnn.py OUT.json -- shared-row screen: how well does one other row of the table predict a row? (CPU, minutes)

For the 4,096 sampled rows (seed 5) the best single parent among all 248,320 rows (self excluded) by |cos| of the raw
rows; the parent with its optimal scale leaves 1 - cos² of the row's energy. Reported: the distribution of cos², the
mean residual share, and the high-rate bits a parent pointer saves, ½·log2(1 / (1 - cos²)) per value, averaged over rows
(an upper bound: it ignores the parent's own quantization noise and the 18-bit pointer). Also the same with the best of
the top-8 parents combined by least squares (a sparse row code, an upper bound for row-level VQ / clustering).
Env E3_CKPT, THREADS."""
import json, math, os, sys
import numpy as np, torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
CK = os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16")
wm = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
NE = "model.language_model.embed_tokens.weight"
rows = np.sort(np.random.default_rng(5).choice(248320, 4096, replace=False))
f = safe_open(f"{CK}/{wm[NE]}", "pt"); sl = f.get_slice(NE)
A = torch.cat([sl[int(i):int(i) + 1] for i in rows]).float(); An = A / A.norm(dim=1, keepdim=True)
best = torch.zeros(len(rows), 8); bidx = torch.zeros(len(rows), 8, dtype=torch.long)
for r0 in range(0, 248320, 8192):
    B = sl[r0:r0 + 8192].float(); Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-30)
    c = (An @ Bn.T).abs()
    self_ = torch.from_numpy(rows)[:, None] == torch.arange(r0, r0 + B.shape[0])[None, :]
    c[self_] = 0
    v, i = c.topk(8, dim=1)
    allv = torch.cat([best, v], 1); alli = torch.cat([bidx, i + r0], 1)
    o = allv.argsort(1, descending=True)[:, :8]; best = allv.gather(1, o); bidx = alli.gather(1, o)
cos2 = best[:, 0].double() ** 2
res8 = []
for k in range(len(rows)):                          # least squares on the top-8 parents
    P = torch.stack([sl[int(j):int(j) + 1][0].float() for j in bidx[k]]).double()
    y = A[k].double(); coef = torch.linalg.lstsq(P.T, y).solution; r = y - P.T @ coef
    res8.append(float(r.pow(2).sum() / y.pow(2).sum()))
res8 = np.array(res8)
out = {"cos2_quantiles": {q: float(np.quantile(cos2.numpy(), q)) for q in (0.5, 0.9, 0.99, 1.0)},
       "resid1_mean": float((1 - cos2).mean()), "bits1_mean": float((0.5 * np.log2(1 / (1 - cos2.numpy()))).mean()),
       "frac_rows_cos2_gt_0.5": float((cos2 > 0.5).double().mean()),
       "resid8_mean": float(res8.mean()), "bits8_mean": float((0.5 * np.log2(1 / res8)).mean()),
       "resid8_quantiles": {q: float(np.quantile(res8, q)) for q in (0.01, 0.1, 0.5)}}
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1], "w"), indent=1)
