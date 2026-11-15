#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sizekl183.py STATE.jsonl RESULTS.jsonl REFS-T183.jsonl OUT.png -- size vs KLD on t183: each size's latest tracked
incumbent (trk183-<tag>) against IQR-V4 and V2 (log-linear interpolation in log KLD between measured sizes); lower panel:
ours ÷ IQR-V4 at the same bpw, and the size IQR-V4 needs for our KLD (bpw saved)."""
import json, math, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
st, res, rf, out = sys.argv[1:5]
order = {}; i = 0
for l in open(st):
    e = json.loads(l)
    if e.get("ev") == "incumbent": order[e["tag"]] = i; i += 1
ours = {}
for l in open(res):
    if '"trk183-' not in l: continue
    r = json.loads(l); t = r["tag"][7:]
    if r.get("kld") is None or t not in order: continue
    s = float(t.split("-")[1])
    if s in (7.0,) or (s in ours and order[ours[s][2]] > order[t]): continue
    ours[s] = (r["bpw"], r["kld"], t)
refs = [json.loads(l) for l in open(rf)]
IQ = sorted((r["bpw"], r["kld"]) for r in refs if r["name"].startswith("iq"))
V2 = sorted((r["bpw"], r["kld"]) for r in refs if r["name"].startswith("v2"))
def ll(P, b):
    for (b0, k0), (b1, k1) in zip(P, P[1:]):
        if b0 <= b <= b1: return math.exp(math.log(k0) + (math.log(k1) - math.log(k0)) * (b - b0) / (b1 - b0))
def inv(P, k):                                                   # bpw at which P reaches KLD k
    for (b0, k0), (b1, k1) in zip(P, P[1:]):
        if k1 <= k <= k0: return b0 + (math.log(k) - math.log(k0)) / (math.log(k1) - math.log(k0)) * (b1 - b0)
O = sorted(ours.values())
fig, (a, b) = plt.subplots(2, 1, figsize=(11, 10), height_ratios=[1.6, 1], sharex=True)
a.plot(*zip(*V2), "-o", color="gray", ms=4, lw=1.2, label="V2 (released)")
a.plot(*zip(*IQ), "-s", color="tab:red", ms=5, lw=1.8, label="IQR-V4")
a.plot([o[0] for o in O], [o[1] for o in O], "-o", color="tab:blue", ms=6, lw=2.2, label="V3 search: latest tracked incumbent per size")
for bp, k, t in O: a.annotate(f"{k:.4f}", (bp, k), textcoords="offset points", xytext=(6, 4), fontsize=8, color="tab:blue")
a.set_yscale("log"); a.grid(alpha=.3, which="both"); a.legend(); a.set_ylabel("mean KLD on t183 (gen rows)")
a.set_title("t183 (fixed tracking slice, never decides): size vs KLD, V3 search incumbents vs IQR-V4", fontsize=11)
rows = []
for bp, k, t in O:
    q = ll(IQ, bp); e = inv(IQ, k)
    if q: rows.append((bp, k / q, (e - bp) if e else None))
b.axhline(1, color="k", lw=1)
b.plot([r[0] for r in rows], [r[1] for r in rows], "-o", color="tab:blue", lw=2)
for bp, r_, d in rows:
    b.annotate(f"{(r_ - 1) * 100:+.0f}%" + (f"\nIQR needs {d:+.2f} bpw" if d is not None else ""), (bp, r_), textcoords="offset points",
               xytext=(5, -24), fontsize=8)
b.set_ylabel("ours ÷ IQR-V4 at the same bpw"); b.set_xlabel("bpw (container bytes × 8 / params)"); b.grid(alpha=.3)
b.set_ylim(min(0.6, min(r[1] for r in rows) * 0.95), max(1.1, max(r[1] for r in rows) * 1.05))
fig.tight_layout(); fig.savefig(out, dpi=110)
for bp, k, t in O: print(f"{bp:.3f} {k:.5f} IQR {ll(IQ, bp) or float('nan'):.5f} {t}")
