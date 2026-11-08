#!/usr/bin/env python3
"""report93.py OUTDIR... -- #39-01 verdict (../w93.md item 2, rule steps 1-5) from settle93 job files
(OUTDIR/LAYER-TENSOR-sS-i8-KK.jsonl). Prints markdown, writes report93.json next to the first OUTDIR."""
import glob, json, math, os, sys
from collections import defaultdict

LEVELS = (2.5, 4.0)
rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/*-i8-K*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
R = defaultdict(dict)                     # (layer, w, sample, K) -> {gain: row} (order / natural arm), "noorder"
for r in rows:
    k = (r["layer"], r["w"], r["sample"], float(r["bits"]))
    R[k]["noorder" if r["arm"] == "noorder" else r["gain"]] = r
gains = sorted({g for v in R.values() for g in v if g != "noorder"})
tensors = sorted({(k[0], k[1]) for k in R})
fam = defaultdict(list)
for l, w in tensors: fam[w].append(l)
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def r(k, m, X):
    v = R.get(k, {})
    return v[m]["rel"][X] / v[1.0]["rel"][X] if m in v and 1.0 in v else None


def cell(ts, Ks, m, X):
    xs = [r((l, w, s, K), m, X) for l, w in ts for s in (0, 1) for K in Ks]
    xs = [x for x in xs if x is not None]
    return geo(xs), len(xs)


out = ["## #39-01 gain on today's stack (held out)", ""]
complete = all(len(R.get((l, w, s, K), {})) >= len(gains) for l, w in tensors for s in (0, 1) for K in LEVELS)
out.append(f"{len(tensors)} tensors, {len(rows)} rows, grid {gains}; complete: {complete}")
# 1. global pick on B1, both K
curve = {m: cell(tensors, LEVELS, m, "B1") for m in gains}
mstar = min(gains, key=lambda m: curve[m][0])
out += ["", "| m | geo r_B1 (both K) | r_B2 K2.5 | r_B2 K4 | r_B (B1+B2) K2.5 | r_B K4 | r_A K2.5 | r_A K4 |", "|---|---|---|---|---|---|---|---|"]
for m in gains:
    out.append(f"| {m:g} | {curve[m][0]:.4f} | " + " | ".join(f"{cell(tensors, (K,), m, X)[0]:.4f}" for X in ("B2", "B", "A") for K in LEVELS) + " |")
res = {"m_star": mstar, "K": {}}
ok = mstar != 1.0
for K in LEVELS:
    g = 1 - cell(tensors, (K,), mstar, "B2")[0]
    agree = sum(geo([x for x in (r((l, w, s, K), mstar, "B2") for s in (0, 1)) if x is not None]) < 1 for l, w in tensors)
    res["K"][K] = {"g": g, "agree": agree}
    ok &= g >= 0.01 and agree >= 18
b2pick = min(gains, key=lambda m: cell(tensors, LEVELS, m, "B2")[0])
edge = mstar in (min(gains), max(gains))
out += ["", f"**Global (rule 1-3):** m* (B1) = {mstar:g}; m* picked on B2 instead: {b2pick:g}. "
        + "; ".join(f"K{K:g}: held-out gain g = {v['g'] * 100:+.2f}%, tensors agreeing {v['agree']} / {len(tensors)}" for K, v in res["K"].items())
        + f". Edge: {edge}. **Verdict: {'PROPOSE m* = %g' % mstar if ok else 'keep 1.0'}**"]
res["verdict"] = f"propose {mstar:g}" if ok else "keep 1.0"; res["edge"] = edge
# 4. per family
out += ["", "**Per family (rule 4, reported):**", "", "| family | K | m*_f (B1) | g_f (B2) | max tensor r_B2 (both samples) | qualifies |", "|---|---|---|---|---|---|"]
res["family"] = {}
for w, ls in fam.items():
    q = []
    for K in LEVELS:
        ts = [(l, w) for l in ls]
        mf = min(gains, key=lambda m: cell(ts, (K,), m, "B1")[0])
        gf = 1 - cell(ts, (K,), mf, "B2")[0]
        mx = max(x for x in (r((l, w, s, K), mf, "B2") for l in ls for s in (0, 1)) if x is not None)
        qual = mf != 1.0 and gf >= 0.01 and mx <= 1.0
        q.append(qual); res["family"].setdefault(w, {})[K] = {"m": mf, "g": gf, "max": mx, "q": qual}
        out.append(f"| {w} | {K:g} | {mf:g} | {gf * 100:+.2f}% | {mx:.4f} | {qual} |")
    res["family"][w]["table"] = all(q)
tab = [w for w in fam if res["family"][w]["table"]]
out.append(f"\nFamilies that would get a table (qualify at both K): {tab or 'none'}")
# reported: #117 order vs natural at m = 1 (readers)
rd = [(l, w) for l, w in tensors if any("noorder" in R.get((l, w, s, K), {}) for s in (0, 1) for K in LEVELS)]
if rd:
    out += ["", "**#117 order vs natural order at m = 1 (readers, reported):** order / natural rel_B"]
    for K in LEVELS:
        xs = [R[(l, w, s, K)][1.0]["rel"]["B"] / R[(l, w, s, K)]["noorder"]["rel"]["B"] for l, w in rd for s in (0, 1)
              if "noorder" in R.get((l, w, s, K), {}) and 1.0 in R[(l, w, s, K)]]
        byf = defaultdict(list)
        for l, w in rd:
            for s in (0, 1):
                v = R.get((l, w, s, K), {})
                if "noorder" in v and 1.0 in v: byf[w].append(v[1.0]["rel"]["B"] / v["noorder"]["rel"]["B"])
        out.append(f"- K{K:g}: geo {geo(xs):.4f} over {len(xs)} ({', '.join(f'{w} {geo(v):.4f}' for w, v in byf.items())}); "
                   f"better in {sum(x < 1 for x in xs)} / {len(xs)}")
# reported: one-sided and no-refit curves
out += ["", "One-sided rel1 (B) and two-sided before the refit (B1+B2 via rel_norefit B1), geo over all, by m:"]
for m in gains:
    o1 = geo([R[k][m]["rel1"]["B"] / R[k][1.0]["rel1"]["B"] for k in R if m in R[k] and 1.0 in R[k]])
    nr = geo([R[k][m]["rel_norefit"]["B1"] / R[k][1.0]["rel_norefit"]["B1"] for k in R if m in R[k] and 1.0 in R[k]])
    out.append(f"- m {m:g}: one-sided {o1:.4f}, before refit (B1) {nr:.4f}")
print("\n".join(out))
json.dump(res, open(os.path.join(sys.argv[1], "..", "report93.json"), "w"), indent=1, default=str)
