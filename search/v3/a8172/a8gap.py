#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""a8gap.py DOCSUMS.json REGISTRY STATE.jsonl OUT.json -- w132 (#172) step 1: the f16-vs-A8 gap of the search incumbents.
DOCSUMS: per-document KL sums (golden gen positions, rxrun.hostd.doc_sums) keyed "SIZE|inc|prev|MODE|GOLDEN[|w132]"
(run/docsums132.py on ds8). Per size and golden: KLD f16 / A8, ratio, paired log ratio (s171b: the search's statistic,
equal-weight mean of the x / c parts, seqrule.Paired) with its delta-method SE and a 2,000-draw per-document bootstrap
95% CI. Ranking: for each pair of incumbents the change of their log KLD ratio under A8 (bootstrap CI, documents paired).
Decision: each incumbent vs the incumbent it replaced, on s171b, f16 vs A8: log r and the accept rule (seqrule.final,
the decision's recorded size offset)."""
import itertools, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import seqrule as SR         # noqa: E402
import slices as SL          # noqa: E402

D = json.load(open(sys.argv[1])); reg = SL.Registry(sys.argv[2]); OUT = sys.argv[4]
PARTS = {"s171b": reg.parts_of("s171b"), "ver3-8k": None}
dec = {}
for l in open(sys.argv[3]):
    e = json.loads(l)
    if e.get("ev") == "decision": dec[e["tag"]] = e
INC = {"2.5": ("v3s171-2.5-explore-00419", "v3s171-2.5-block-00413"), "3.5": ("v3s171-3.5-block-00483", "v3s171-3.5-pair-00396"),
       "5.0": ("v3s171-5-pair-00449", "v3s171-5-pair-00409"), "6.0": ("v3s171-6-explore-00372", "v3s171-6-inc-00358")}
rng = np.random.default_rng(172); NB = 2000


def get(size, who, mode, g):
    k = f"{size}|{who}|{mode}|{g}"
    return D.get(k + "|w132") if mode == "a8" else D.get(k)


def arr(e): return np.array([d[0] for d in e["docs"]]), np.array([d[1] for d in e["docs"]])


def stat(A, B, parts, idx=None):
    """the search's log ratio of B over A (pmean over parts, pooled for one-part goldens) on documents idx"""
    idx = np.arange(len(A)) if idx is None else idx
    if parts is None: return math.log(B[idx].sum() / A[idx].sum())
    P = np.array(parts)[idx]
    return float(np.mean([math.log(B[idx][P == p].sum() / A[idx][P == p].sum()) for p in sorted(set(parts))]))


def boot(fn, n, parts):
    """2,000 resamples of the documents (stratified by part when the golden has parts) -> 2.5 / 97.5 %"""
    if parts is None: draws = [rng.integers(0, n, n) for _ in range(NB)]
    else:
        P = np.array(parts); grp = [np.where(P == p)[0] for p in sorted(set(parts))]
        draws = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp]) for _ in range(NB)]
    v = np.array([fn(ix) for ix in draws])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(v.std())


def paired_se(A, B, parts):
    P = SR.Paired(len(A), {"kind": "full", "stat": "pmean" if parts else "pooled"})
    for i in range(len(A)): P.add(float(A[i]), float(B[i]), parts[i] if parts else "all")
    return P


out = {"gap": {}, "rank": {}, "decision": {}, "timing": {}}
for g in ("ver3-8k", "s171b"):
    parts = PARTS[g]
    for s in INC:
        f, a = get(s, "inc", "f16", g), get(s, "inc", "a8", g)
        if not (f and a): continue
        A, n = arr(f); B, n2 = arr(a); assert (n == n2).all()
        P = paired_se(A, B, parts); lo, hi, sd = boot(lambda ix: stat(A, B, parts, ix), len(A), parts)
        o = {"kld_f16": f["kld"], "kld_a8": a["kld"], "ratio_pooled": a["kld"] / f["kld"], "logr": P.d(), "ratio": math.exp(P.d()),
             "se": P.se(), "ci95": [lo, hi], "positions": f["positions"], "all_f16": f.get("all_kld"), "all_a8": a.get("all_kld")}
        if parts: o["parts"] = {p: math.exp(stat(A, B, [p] * len(A), np.where(np.array(parts) == p)[0])) for p in sorted(set(parts))}
        out["gap"][f"{s}|{g}"] = o
        if a.get("docs_s"): out["timing"][f"{s}|{g}"] = {"a8_s": a.get("docs_s"), "f16_s": f.get("docs_s")}
    for s1, s2 in itertools.combinations(INC, 2):
        e = [get(s, "inc", m, g) for s in (s1, s2) for m in ("f16", "a8")]
        if not all(e): continue
        F1, A1, F2, A2 = (arr(x)[0] for x in e)
        rf = stat(F1, F2, parts); ra = stat(A1, A2, parts)
        lo, hi, sd = boot(lambda ix: stat(A1, A2, parts, ix) - stat(F1, F2, parts, ix), len(F1), parts)
        out["rank"][f"{s1}->{s2}|{g}"] = {"logr_f16": rf, "logr_a8": ra, "diff": ra - rf, "ci95": [lo, hi], "sd": sd}
for s, (t, prev) in INC.items():
    g = "s171b"; parts = PARTS[g]; e = [get(s, w, m, g) for w in ("prev", "inc") for m in ("f16", "a8")]
    if not all(e): continue
    PF, PA, IF, IA = (arr(x)[0] for x in e); off = dec[t].get("off", 0.0) or 0.0
    res = {}
    for m, (A, B) in (("f16", (PF, IF)), ("a8", (PA, IA))):
        P = SR.Paired(len(A), {"kind": "full", "stat": "pmean", "off": off})
        for i in range(len(A)): P.add(float(A[i]), float(B[i]), parts[i])
        res[m] = {"logr": P.d(), "da": P.da(), "z": P.z(), "se": P.se(), "decision": P.final(), "parts": {k: v["logr"] for k, v in P.summary()["parts"].items()}}
    lo, hi, sd = boot(lambda ix: stat(PA, IA, parts, ix) - stat(PF, IF, parts, ix), len(PF), parts)
    res["diff"] = res["a8"]["logr"] - res["f16"]["logr"]; res["diff_ci95"] = [lo, hi]; res["recorded"] = {k: dec[t].get(k) for k in ("decision", "logr", "z", "off")}
    out["decision"][f"{s}: {prev} -> {t}"] = res
json.dump(out, open(OUT, "w"), indent=1)
print("gap (A8 / f16, search statistic; s171b = mean of the parts' log ratios):")
for k, o in out["gap"].items():
    print(f"  {k:14s} f16 {o['kld_f16']:.6f}  A8 {o['kld_a8']:.6f}  ratio {o['ratio']:.4f} [{math.exp(o['ci95'][0]):.4f}, {math.exp(o['ci95'][1]):.4f}]"
          f"  (all-position {o['all_f16']:.5f} -> {o['all_a8']:.5f}: {o['all_a8'] / o['all_f16']:.4f})" + (f"  parts {', '.join(f'{p} {v:.4f}' for p, v in o['parts'].items())}" if o.get("parts") else ""))
print("ranking (log KLD ratio between incumbents, A8 minus f16):")
for k, o in out["rank"].items(): print(f"  {k:20s} f16 {o['logr_f16']:+.4f}  A8 {o['logr_a8']:+.4f}  diff {o['diff']:+.4f} [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]")
print("decisions on s171b (prev -> incumbent):")
for k, o in out["decision"].items():
    print(f"  {k}: f16 log r {o['f16']['logr']:+.4f} z {o['f16']['z']:+.2f} {o['f16']['decision']} | A8 log r {o['a8']['logr']:+.4f} z {o['a8']['z']:+.2f} {o['a8']['decision']}"
          f" | diff {o['diff']:+.4f} [{o['diff_ci95'][0]:+.4f}, {o['diff_ci95'][1]:+.4f}] | recorded {o['recorded']['decision']} {o['recorded']['logr']:+.4f}")
print("timing (radiance 'N documents in S s', A8 / f16 on different cards):", {k: v for k, v in out["timing"].items()})
