#!/usr/bin/env python3
"""report91.py OUTDIR -- ../w91.md item B6's rule over settle91.py's outputs (OUTDIR/k1, OUTDIR/k05). r(t, s, l, n) =
rel_X(sigma_eff[n]) / rel_X(sigma_eff[mean]) (two-sided, X = B1 / B2 held out, A in-sample); pick n* in {med, trl} on
B1 over the 7 tensors x 2 samples x {i8 K2.5, i8 K3}; judge on B2: g = 1 - geo r_B2(n*), a tensor agrees if its geo
r_B2(n*) < 1; WIN iff g >= 2% and >= 5 of 7 agree. Reported: early / late subsets, each level, both alternatives, A,
the i4 K1 / K0.5 cells. Writes OUTDIR/report91.json and prints a summary."""
import glob, json, math, os, sys
from collections import defaultdict

O = sys.argv[1]
EARLY = [(2, "ssm_qkv"), (3, "attn_qg"), (7, "attn_v"), (10, "ssm_z"), (11, "attn_k")]; LATE = [(50, "ssm_qkv"), (63, "attn_k")]
TS = EARLY + LATE
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")
R = defaultdict(dict)       # (layer, tensor, sample, cb, K) -> {norm: rel dict}
ratio = {}
for sub in ("k1", "k05"):
    for sj in glob.glob(f"{O}/{sub}/*.sig.json"):
        st = os.path.basename(sj)[:-len(".sig.json")]; l, t, s = st.split("-"); l, s = int(l), int(s[1:])
        sig = json.load(open(sj))["sigma_eff"]
        for n, v in sig.items():
            f = f"{O}/{sub}/{st}-d{v:g}.jsonl"
            if not os.path.exists(f): continue
            for line in open(f):
                o = json.loads(line); R[(l, t, s, o["cb"], o["bits"])][n] = o["rel"]
        N = json.load(open(f"{O}/{st}.norm.json")); ratio[(l, t)] = (N["med"] / N["mean"], N["trl"] / N["mean"])


def r(key, n, X): return R[key][n][X] / R[key]["mean"][X]


def cells(levels, ts, n, X):
    return [r(k, n, X) for k in R if (k[0], k[1]) in ts and (k[3], k[4]) in levels and n in R[k] and "mean" in R[k]]


DEC = [("i8", 2.5), ("i8", 3.0)]
out = {"n_cells": len(R)}
missing = [(l, t, s, cb, K) for (l, t) in TS for s in (0, 1) for cb, K in DEC if not {"mean", "med", "trl"} <= set(R.get((l, t, s, cb, K), {}))]
out["missing"] = missing
b1 = {n: geo(cells(DEC, TS, n, "B1")) for n in ("med", "trl")}
ns = min(b1, key=b1.get)
gB2 = geo(cells(DEC, TS, ns, "B2")); g = 1 - gB2
agree = {f"{l}:{t}": geo([r((l, t, s, cb, K), ns, "B2") for s in (0, 1) for cb, K in DEC if {"mean", ns} <= set(R.get((l, t, s, cb, K), {}))]) for l, t in TS}
na = sum(v < 1 for v in agree.values())
win = (not missing) and g >= 0.02 and na >= 5
out.update({"b1_geo": b1, "n_star": ns, "g_B2": g, "agree": agree, "n_agree": na, "verdict": "WIN" if win else "mean normalization stands"})
rep = {}
for n in ("med", "trl"):
    for name, ts in (("all7", TS), ("early5", EARLY), ("late2", LATE)):
        for lv in (("dec", DEC), ("i8K2.5", [("i8", 2.5)]), ("i8K3", [("i8", 3.0)]), ("i4K1", [("i4", 1.0)]), ("i4K0.5", [("i4", 0.5)])):
            rep[f"{n}/{name}/{lv[0]}"] = {X: geo(cells(lv[1], ts, n, X)) for X in ("B1", "B2", "B", "A")}
out["report"] = rep
out["sigma_ratio"] = {f"{l}:{t}": v for (l, t), v in ratio.items()}
json.dump(out, open(f"{O}/report91.json", "w"), indent=1)
print(f"missing cells: {len(missing)}")
print("effective sigma ratio N/N_mean (med, trl): " + ", ".join(f"{l}:{t} {a:.3f}/{b:.3f}" for (l, t), (a, b) in sorted(ratio.items())))
for k, v in rep.items(): print(f"{k:22s} " + " ".join(f"{X} {v[X]:.4f}" for X in ("B1", "B2", "B", "A")))
print("tensor agree (geo r_B2 of n*): " + ", ".join(f"{k} {v:.4f}" for k, v in agree.items()))
print(f"B6: n* = {ns} (B1 geo med {b1['med']:.4f}, trl {b1['trl']:.4f}); g_B2 = {g * 100:+.2f}%, {na}/7 agree -> {out['verdict']}")
