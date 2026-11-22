#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""lowrank218.py RDIR [--ranks 4,16] [--sub 2] -- #218 follow-up (11-22): the low-rank-projected code-QAD direction. Per probe tensor, from
round RDIR's three 100k-row chunk gradients g0, g1, g2 (same codes, disjoint documents): U, V = the top-r singular vectors of the
subspace chunk g<sub> (randomized SVD, q = r + 8, 6 power iterations, seed 0), direction D_r = U U^T ((ga + gb) / 2) V V^T from the
OTHER two chunks (no chunk both picks the subspace and gives the signal). Writes RDIR/lr<r>/<key>.g + grad.json (encode218.py's grad
format; g_rms = rms(D_r)) and RDIR/lowrank.json: per tensor and r, the raw cosine cos(ga, gb), the projected cosine cos(P ga, P gb)
and the projection's energy share |P ga|^2 / |ga|^2 (the structure check). On the card when one is visible."""
import argparse, json, os

import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument("rdir"); ap.add_argument("--ranks", default="4,16"); ap.add_argument("--sub", type=int, default=2)
a = ap.parse_args()
dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
ranks = [int(x) for x in a.ranks.split(",")]
other = [c for c in (0, 1, 2) if c != a.sub]
zs = {c: json.load(open(f"{a.rdir}/g{c}/grad.json")) for c in (0, 1, 2)}
meta = zs[0]["tensors"]
for r in ranks: os.makedirs(f"{a.rdir}/lr{r}", exist_ok=True)
def cos(x, y): return float((x * y).sum() / (x.norm() * y.norm()))
out = {}
for key in sorted(meta):
    m = meta[key]
    ld = lambda c: torch.from_numpy(np.fromfile(f"{a.rdir}/g{c}/{key}.g", np.float32).reshape(m["k"], m["n"])).to(dev)
    gs, ga, gb = ld(a.sub), ld(other[0]), ld(other[1])
    rec = {"raw_cos": cos(ga, gb)}
    torch.manual_seed(0)
    U0, S0, V0 = torch.svd_lowrank(gs, q=max(ranks) + 8, niter=6)
    for r in ranks:
        U, V = U0[:, :r], V0[:, :r]
        P = lambda g: U @ ((U.T @ g) @ V) @ V.T
        pa, pb = P(ga), P(gb)
        D = (pa + pb) / 2
        rec[f"r{r}"] = {"proj_cos": cos(pa, pb), "energy": float(pa.norm() ** 2 / ga.norm() ** 2), "sv_share": float((S0[:r] ** 2).sum() / gs.norm() ** 2)}
        D.float().cpu().numpy().tofile(f"{a.rdir}/lr{r}/{key}.g")
        rec[f"r{r}"]["d_rms"] = float(D.pow(2).mean().sqrt())
    out[key] = rec
    print(f"{key.split('layers.')[1]:30s} raw {rec['raw_cos']:.3f}  " + "  ".join(
        f"r{r} cos {rec[f'r{r}']['proj_cos']:.3f} energy {rec[f'r{r}']['energy']:.3f}" for r in ranks), flush=True)
    del gs, ga, gb
for r in ranks:
    t = {k: {**meta[k], "g_rms": out[k][f"r{r}"]["d_rms"]} for k in meta}
    json.dump({"rows": sum(zs[c]["rows"] for c in other), "codes": zs[0]["codes"], "lowrank": {"r": r, "sub": a.sub, "signal": other},
               "tensors": t}, open(f"{a.rdir}/lr{r}/grad.json", "w"))
json.dump(out, open(f"{a.rdir}/lowrank.json", "w"), indent=1)
