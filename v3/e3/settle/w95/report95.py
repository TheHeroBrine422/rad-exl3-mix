#!/usr/bin/env python3
"""report95.py OUTDIR... [--w87 DIR] -- #126's rule (../w95.md): gate G0 (the recomputed i4 rows == w87's), then
R(K) = geo mean over tensors x samples of rel_B(i8) / rel_B(i4) at K0.125 / 0.5 / 1, with B1 / B2 / A / one-sided /
per-group / per-family breakdowns and a tensor-bootstrap 95% interval. OUTDIR: settle95's out dirs (k<K>/ inside),
one per box. Markdown on stdout."""
import argparse, glob, json, math, os, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="+")
ap.add_argument("--w87", default=os.path.join(HERE, "..", "data", "w87", "out"))
a = ap.parse_args()
KS = (0.125, 0.5, 1.0)


def rows(paths):
    for p in paths:
        for line in open(p):
            if line.strip(): yield json.loads(line)


key = lambda r: (r["layer"], r["w"], r["sample"], float(r["bits"]), r["sigma"])
new = defaultdict(dict)                               # (layer, w, sample, K, sigma) -> {cb: row}
for d in a.out:
    for r in rows(glob.glob(os.path.join(d, "k*", "*-d*.jsonl"))):
        new[key(r)][r["cb"]] = r
old = {}
for r in rows(glob.glob(os.path.join(a.w87, "*", "*-d*.jsonl"))):
    if r["cb"] == "i4" and float(r["bits"]) in KS: old[key(r)] = r

geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
print("# #126 i8 vs i4 at the sub-1 levels (w87 two-sided held-out proxy)\n")
pairs = {k: v for k, v in new.items() if "i4" in v and "i8" in v}
print(f"pairs: {len(pairs)} (expected 27 x 2 x 3 = 162)\n")

# ---- G0
dev, miss = [], 0
for k, v in pairs.items():
    o = old.get(k)
    if o is None: miss += 1; continue
    dev.append(abs(v["i4"]["rel"]["B"] / o["rel"]["B"] - 1))
g0 = not miss and dev and max(dev) <= 1e-6
print(f"## G0: i4 rows vs w87's\n\n{len(dev)} compared, {miss} without a w87 row, max |rel_B ratio - 1| = "
      f"{max(dev) if dev else float('nan'):.2e} -> **{'PASS' if g0 else 'FAIL'}**\n")

# ---- R(K)
def ratio(v, X, f="rel"): return v["i8"][f][X] / v["i4"][f][X]


def boot(per_t, n=2000, seed=126):
    ts = sorted(per_t); rng = random.Random(seed); out = []
    for _ in range(n):
        s = [x for t in (rng.choice(ts) for _ in ts) for x in per_t[t]]
        out.append(geo(s))
    out.sort(); return out[int(0.025 * n)], out[int(0.975 * n) - 1]


print("## R(K) = geo mean rel(i8) / rel(i4), 27 tensors x 2 samples\n")
print("| K | n | **B (held-out)** | 95% CI (tensor bootstrap) | B1 | B2 | A (in-sample) | B one-sided | inputs B | rest B | max tensor B | s_exl i8/i4 | verdict |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
verdict = {}
for K in KS:
    vs = {k: v for k, v in pairs.items() if k[3] == K}
    if not vs: print(f"| {K:g} | 0 | | | | | | | | | | | missing |"); continue
    rb = [ratio(v, "B") for v in vs.values()]
    per_t = defaultdict(list)
    for k, v in vs.items(): per_t[(k[0], k[1])].append(ratio(v, "B"))
    lo, hi = boot(per_t)
    grp = lambda g: geo([ratio(v, "B") for v in vs.values() if v["i4"]["group"] == g])
    mt = max(per_t.items(), key=lambda kv: geo(kv[1]))
    R = geo(rb); verdict[K] = R
    print(f"| {K:g} | {len(vs)} | **{R:.4f}** | {lo:.4f}–{hi:.4f} | {geo([ratio(v, 'B1') for v in vs.values()]):.4f} | "
          f"{geo([ratio(v, 'B2') for v in vs.values()]):.4f} | {geo([ratio(v, 'A') for v in vs.values()]):.4f} | "
          f"{geo([ratio(v, 'B', 'rel1') for v in vs.values()]):.4f} | {grp('input'):.4f} | {grp('rest'):.4f} | "
          f"{geo(mt[1]):.4f} ({mt[0][0]}:{mt[0][1]}) | {geo([v['i8']['s_exl'] / v['i4']['s_exl'] for v in vs.values()]):.3f} | "
          f"{'i8 > 1% worse' if R > 1.01 else 'ok'} |")
print()
print("## per family (B, geo over layers x samples)\n")
fams = sorted({k[1] for k in pairs})
print("| family | " + " | ".join(f"K{K:g}" for K in KS) + " |")
print("|---|" + "---|" * len(KS))
for f in fams:
    print(f"| {f} | " + " | ".join(f"{geo([ratio(v, 'B') for k, v in pairs.items() if k[1] == f and k[3] == K]):.4f}" for K in KS) + " |")
print("\n## absolute held-out rel_B (geo), i4 / i8\n")
print("| K | i4 | i8 |\n|---|---|---|")
for K in KS:
    vs = [v for k, v in pairs.items() if k[3] == K]
    if vs: print(f"| {K:g} | {geo([v['i4']['rel']['B'] for v in vs]):.4f} | {geo([v['i8']['rel']['B'] for v in vs]):.4f} |")
done = len(verdict) == 3 and all(len([k for k in pairs if k[3] == K]) == 54 for K in KS)
fail = [K for K, R in verdict.items() if R > 1.01]
print(f"\n**Rule:** {'incomplete' if not done else ('G0 FAIL: ratios not read' if not g0 else ('i4 stays at K ' + ', '.join(f'{K:g}' for K in fail) + ' (i8 > 1% worse)' if fail else 'PASS: switch to i8 at K0.125-1'))}")
