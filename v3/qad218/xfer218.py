#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""xfer218.py OUT.json --chunks D0,D1,... --check GT [--splits NAME:A:B;...] [--rank 4] -- #218 scaling follow-up (11-22): does the
low-rank direction transfer to held-out documents better with more rows? CPU, no card.

Chunks = 100k-row training gradients dL/dW_hat at the released codes (probe218.py grad); GT = a check gradient on held-out tune documents.
Per split (A = the chunks that pick the subspace, B = the signal chunks; rows-weighted means G_A, G_B), per probe tensor:
  weights  U, V = top-r singular vectors of G_A; D = U U^T G_B V V^T; the step W_hat - eta delta D / rms(D) (encode218's units, delta from
           runs/qerr.json); its slope dKL/deta under a gradient G is -delta / rms(D) * <G, D>. Reported for G = G_B (in-sample: the signal
           itself), G = G_A (the subspace picker), G = GT (held-out). held-out fraction = slope(GT) / slope(G_B).
  scales   log-multiplier gradients of a weight gradient X (r on suh, c on svh; W = diag(suh) Hb W_hat Hb diag(svh)):
           g_r = rowsum(Hk X o Hk W_hat) [k], g_c = colsum(W_hat Hn o X Hn) [n]. Two scale steps, each unit-norm per tensor (r and c
           together): "proj" = the scale gradient of D (what the low-rank weight direction asks of the scales), "plain" = the scale gradient
           of G_B (no projection); slope under X = -<scale grad of X, s>. Same in-sample / picker / held-out comparison.
Totals sum the per-tensor slopes (each tensor's step normalised on its own, as encode218 --step quant)."""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
from rxq import deq  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--chunks", required=True); ap.add_argument("--check", required=True)
ap.add_argument("--splits", default="s300:2:0,1;s100:0:5;s500:0,1,2,3,4:5,6,7,8,9"); ap.add_argument("--rank", type=int, default=4)
ap.add_argument("--norm-chunks", action="store_true")
ap.add_argument("--lv", default="/data/w218/lv/2.50/trunk"); ap.add_argument("--qerr", default="/data/w218/runs/qerr.json")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
chunks = a.chunks.split(",")
zs = [json.load(open(f"{d}/grad.json")) for d in chunks]
assert all(z["codes"] == "" for z in zs), "every chunk at the released codes"
meta = zs[0]["tensors"]; qe = json.load(open(a.qerr))
splits = []
for s in a.splits.split(";"):
    n, A, B = s.split(":"); A = [int(x) for x in A.split(",")]; B = [int(x) for x in B.split(",")]
    if max(A + B) < len(chunks): splits.append((n, A, B))
lev = deq.Level(a.lv, "cpu"); table = torch.from_numpy(deq.lut(lev.codebook)); Hb = deq.hadamard128("cpu")


def ld(d, k):
    m = meta[k]; return torch.from_numpy(np.fromfile(f"{d}/{k}.g", np.float32).reshape(m["k"], m["n"]))


def mean(idx, k):
    """rows-weighted mean of the chunks; --norm-chunks: each chunk scaled to unit norm first (the chunk gradients' rms varies ~35x:
    a few documents dominate a chunk; this weights chunks equally by direction)"""
    N = sum(zs[i]["rows"] for i in idx); g = None
    for i in idx:
        x = ld(chunks[i], k)
        x = x / x.norm() if a.norm_chunks else x * (zs[i]["rows"] / N)
        g = x if g is None else g + x
    return g


def hk(x): kk, n = x.shape; return (Hb @ x.reshape(kk // 128, 128, n)).reshape(kk, n)
def hn(x): kk, n = x.shape; return (x.reshape(kk, n // 128, 128) @ Hb).reshape(kk, n)


def sgrad(X, What, HkW, WHn):
    return torch.cat([(hk(X) * HkW).sum(1), (WHn * hn(X)).sum(0)])


out = {}
for key in sorted(meta):
    m = meta[key]; kk, n = m["k"], m["n"]
    raw = lev.raw(key); W = deq.decode_hat(raw["trellis"], raw["trellis"].shape[2] / 16, table)
    HkW, WHn = hk(W), hn(W)
    GT = ld(a.check, key); sT = sgrad(GT, W, HkW, WHn)
    rec = {}
    for name, A, B in splits:
        GA, GB = mean(A, key), mean(B, key)
        torch.manual_seed(0)
        U, S, V = torch.svd_lowrank(GA, q=a.rank + 8, niter=6); U, V = U[:, :a.rank], V[:, :a.rank]
        D = U @ ((U.T @ GB) @ V) @ V.T
        unit = qe[key]["delta"] / float(D.pow(2).mean().sqrt())
        w = {g: -unit * float((X * D).sum()) for g, X in (("in", GB), ("picker", GA), ("held", GT))}
        sA, sB, sD = sgrad(GA, W, HkW, WHn), sgrad(GB, W, HkW, WHn), sgrad(D, W, HkW, WHn)
        sc = {}
        for sn, step in (("proj", sD), ("plain", sB)):
            u = step / step.norm()
            sc[sn] = {g: -float(x @ u) for g, x in (("in", sB), ("picker", sA), ("held", sT))}
        rec[name] = {"w": w, "s": sc}
        del GA, GB, D
    out[key] = rec
    print(key.split("layers.")[1], json.dumps({n: {"w": {g: round(v, 4) for g, v in r["w"].items()}} for n, r in rec.items()}), flush=True)
tot = {}
for n, _, _ in splits:
    t = {"w": {}, "s_proj": {}, "s_plain": {}}
    for k, r in out.items():
        for g in ("in", "picker", "held"):
            t["w"][g] = t["w"].get(g, 0) + r[n]["w"][g]
            t["s_proj"][g] = t["s_proj"].get(g, 0) + r[n]["s"]["proj"][g]
            t["s_plain"][g] = t["s_plain"].get(g, 0) + r[n]["s"]["plain"][g]
    for x in t.values(): x["held_fraction"] = x["held"] / x["in"]
    tot[n] = t
    print(f"TOTAL {n}: " + "  ".join(f"{x} in {v['in']:+.4g} picker {v['picker']:+.4g} held {v['held']:+.4g} (held/in {v['held_fraction']:+.3f})" for x, v in t.items()), flush=True)
json.dump({"chunks": chunks, "check": a.check, "rank": a.rank, "totals": tot, "tensors": out}, open(a.out, "w"), indent=1)
