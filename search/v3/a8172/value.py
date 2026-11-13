#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""value.py DOCSUMS.json REGISTRY STATE.jsonl OUT.json -- RULES-value.md (w132, #172): initial pick vs current
incumbent per size, A8, on s171c + s171d pooled by part. Primary: size-adjusted pmean D_s = d_s + b_s (bpw_cur -
bpw_init), b_s = the driver's last b; pooled = mean of D_s over the sizes; 95% CIs from 2,000 document bootstrap
resamples stratified by part, one resample shared by all sizes."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import slices as SL          # noqa: E402

D = json.load(open(sys.argv[1])); reg = SL.Registry(sys.argv[2]); OUT = sys.argv[4]
SLICES = ("s171c", "s171d")
first, cur, sub, bb = {}, {}, {}, {}
for l in open(sys.argv[3]):
    e = json.loads(l)
    if e["ev"] == "incumbent": first.setdefault(e["size"], e["tag"]); cur[e["size"]] = e["tag"]
    elif e["ev"] == "submit": sub[e["tag"]] = e
    elif e["ev"] == "b": bb[e["size"]] = e["b"]
SIZES = [1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
parts = np.array([p for sl in SLICES for p in reg.parts_of(sl)])
slc = np.array([sl for sl in SLICES for _ in reg.parts_of(sl)])
P = sorted(set(parts.tolist())); grp = [np.where(parts == p)[0] for p in P]


def vec(t):
    a = [D[f"{t}|{sl}"]["docs"] for sl in SLICES]
    return np.array([d[0] for x in a for d in x])


def pmean(A, B, idx):
    pi = parts[idx]
    return float(np.mean([math.log(B[idx][pi == p].sum() / A[idx][pi == p].sum()) for p in P]))


rng = np.random.default_rng(172); NB = 2000
draws = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp]) for _ in range(NB)]
allix = np.arange(len(parts))
out = {"sizes": {}, "pooled": None, "n_docs": {p: int((parts == p).sum()) for p in P}}
boots = []
for s in SIZES:
    ti, tc = first[s], cur[s]
    A, B = vec(ti), vec(tc); off = bb[s] * (sub[tc]["bpw"] - sub[ti]["bpw"])
    d = pmean(A, B, allix); bs = np.array([pmean(A, B, ix) for ix in draws]) + off; boots.append(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    o = {"initial": ti, "current": tc, "bpw_initial": sub[ti]["bpw"], "bpw_current": sub[tc]["bpw"], "b": bb[s], "off": off,
         "d_raw": d, "D": d + off, "ci95": [float(lo), float(hi)], "reading": "gained" if hi < 0 else "lost" if lo > 0 else "null",
         "kld_initial": {sl: D[f"{ti}|{sl}"]["kld"] for sl in SLICES}, "kld_current": {sl: D[f"{tc}|{sl}"]["kld"] for sl in SLICES},
         "per_slice_D": {sl: pmean(A, B, np.where(slc == sl)[0]) + off for sl in SLICES},
         "per_part_raw": {p: math.log(B[parts == p].sum() / A[parts == p].sum()) for p in P}}
    out["sizes"][str(s)] = o
pool = np.mean(boots, axis=0); Dp = float(np.mean([out["sizes"][str(s)]["D"] for s in SIZES])); lo, hi = np.percentile(pool, [2.5, 97.5])
out["pooled"] = {"D": Dp, "ci95": [float(lo), float(hi)], "reading": "gained" if hi < 0 else "lost" if lo > 0 else "null"}
json.dump(out, open(OUT, "w"), indent=1)
print(f"docs: {out['n_docs']} (s171c + s171d)")
print(f"{'size':>5} {'D (size-adj)':>13} {'95% CI':>20} {'reading':>8} {'raw d':>8} {'s171c':>8} {'s171d':>8} {'c raw':>8} {'x raw':>8}  KLD init -> cur (c / d)")
for s in SIZES:
    o = out["sizes"][str(s)]
    print(f"{s:5.2f} {o['D']:+13.4f} [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] {o['reading']:>8} {o['d_raw']:+8.4f} {o['per_slice_D']['s171c']:+8.4f} {o['per_slice_D']['s171d']:+8.4f}"
          f" {o['per_part_raw']['c']:+8.4f} {o['per_part_raw']['x']:+8.4f}  {o['kld_initial']['s171c']:.5f}/{o['kld_initial']['s171d']:.5f} -> {o['kld_current']['s171c']:.5f}/{o['kld_current']['s171d']:.5f}")
p = out["pooled"]; print(f"pooled (mean over {len(SIZES)} sizes): D {p['D']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] -> {p['reading']}  (x{math.exp(p['D']):.4f})")
