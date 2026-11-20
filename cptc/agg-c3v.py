#!/usr/bin/env python3
"""agg-c3v.py C3V.jsonl -- C3 v2 table: global CPTC tables vs mad / mul1 per tensor and K (rel = H-weighted relative
error, same rows / rotation / LDLQ for every arm). Per K: geometric-mean ratio over tensors with a bootstrap 95 % CI
over tensors, and the mad-vs-mul1 spread as the empirical scale of 'two equal-quality codebooks' noise."""
import json, sys, math, numpy as np
R = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
M = {}
for r in R:                       # merge arm files (base / cptc rows of the same tensor): per K dict union
    if "per_K" not in r: continue
    m = M.setdefault((r["layer"], r["w"]), {"layer": r["layer"], "w": r["w"], "per_K": {}})
    for k, d in r["per_K"].items(): m["per_K"].setdefault(k, {}).update(d)
R = [m for m in M.values() if all(f"cptc{k}" in d and "mad" in d for k, d in m["per_K"].items())]
Ks = sorted({int(k) for r in R for k in r["per_K"]})
def ci(v, n=4000, rng=np.random.default_rng(0)):
    v = np.asarray(v); b = [np.mean(rng.choice(v, len(v))) for _ in range(n)]
    return np.mean(v), np.percentile(b, 2.5), np.percentile(b, 97.5)
pc = lambda x: f"{100 * (math.exp(x) - 1):+.1f}%"
print("| K | n | CPTC vs mad (geo-mean, 95% CI) | CPTC vs mul1 | mad vs mul1 | CPTC wins vs mad |\n|---|---|---|---|---|---|")
for K in Ks:
    rows = [r["per_K"][str(K)] for r in R if str(K) in r["per_K"]]
    cm = [math.log(d[f"cptc{K}"] / d["mad"]) for d in rows]; cu = [math.log(d[f"cptc{K}"] / d["mul1"]) for d in rows]
    mu = [math.log(d["mad"] / d["mul1"]) for d in rows]
    a, b, c = ci(cm); e = ci(cu); g = ci(mu)
    print(f"| {K} | {len(rows)} | {pc(a)} [{pc(b)}, {pc(c)}] | {pc(e[0])} [{pc(e[1])}, {pc(e[2])}] | {pc(g[0])} [{pc(g[1])}, {pc(g[2])}] | {sum(x < 0 for x in cm)}/{len(cm)} |")
if True:   # per-tensor detail
    print("\nper tensor (CPTC / mad - 1):")
    for r in sorted(R, key=lambda r: (r["w"], r["layer"])):
        print(f"  {r['layer']:>2}/{r['w']:<12}", " ".join(f"K{k} {100 * (r['per_K'][k][f'cptc{k}'] / r['per_K'][k]['mad'] - 1):+5.1f}%" for k in sorted(r["per_K"], key=int)))
