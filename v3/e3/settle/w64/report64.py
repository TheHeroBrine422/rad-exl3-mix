#!/usr/bin/env python3
"""report64.py DATADIR -- w64's CPU readouts and verdicts (../w64.md items 0, 2, 4) from the boxes' outputs copied to
DATADIR/<box>/out/<mode>/<cb>/s<seed>/<layer>-<tensor>-K<k>.jsonl (box = cpu1, cpu2). Markdown on stdout.
w43's forward-order cd data (item 0b comparison) is read from ../data/v2/<box>/cd/."""
import glob, json, math, os, sys
import numpy as np

D = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}                                                     # (mode, cb, seed, layer, w, K) -> record
for f in glob.glob(f"{D}/*/out/*/*/s*/*.jsonl"):
    p = f.split("/"); mode, cb, seed = p[-4], p[-3], int(p[-2][1:])
    for l in open(f):
        o = json.loads(l); R[(mode, cb, seed, int(o["layer"]), o["w"], float(o["bits"]))] = o
geo = lambda v: float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")
L = [f"# w64 CPU readout (report64.py, {len(R)} jobs)", ""]

# ---- item 0 -----------------------------------------------------------------------------------------------------------
GPU = {(3, 2): (0.9838, 0.9894), (3, 3): (0.9850, 0.9900), (3, 4): (0.9853, 0.9899),
       (19, 2): (0.9937, 0.9942), (19, 3): (0.9958, 0.9960), (19, 4): (0.9958, 0.9952),
       (43, 2): (0.9954, 0.9948), (43, 3): (0.9951, 0.9957), (43, 4): (0.9961, 0.9958)}
L += ["## Item 0 · settle64 (exllamav3's LDLQ order) vs w50's GPU tile-CD ratios (attn_qg, same blocks)", "",
      "cd3 ÷ LDLQ (rel after the refit), mul1x_i8. Rule: geo over the two samples within ±0.5% of the GPU's.", "",
      "| layer | K | CPU s0 / s1 | CPU geo | GPU s0 / s1 | GPU geo | CPU − GPU | ok |", "|---|---|---|---|---|---|---|---|"]
n0 = ok0 = 0
for (l, K), g in GPU.items():
    c = [R.get(("cd", "mul1x_i8", s, l, "attn_qg", float(K))) for s in (0, 1)]
    if None in c: L.append(f"| {l} | {K} | pending | | {g[0]} / {g[1]} | {geo(g):.4f} | | |"); continue
    r = [o["rel"]["cd3"] / o["rel"]["ldlq"] for o in c]; d = geo(r) / geo(g) - 1; ok = abs(d) <= 0.005
    n0 += 1; ok0 += ok
    L.append(f"| {l} | {K} | {r[0]:.4f} / {r[1]:.4f} | {geo(r):.4f} | {g[0]} / {g[1]} | {geo(g):.4f} | {100 * d:+.2f}% | {'yes' if ok else '**no**'} |")
L += ["", f"**Item 0: {'PASS' if ok0 == 9 else 'FAIL' if n0 == 9 else 'pending'}** ({ok0} of {n0} within ±0.5%; 9 needed)", ""]
# 0b: rev vs w43's fwd, same codebook (mul1_i8) and blocks
L += ["Order alone (K2, mul1_i8, LDLQ error after the refit): rev (settle64) ÷ fwd (w43's settle.py run)", "",
      "| layer | s0 | s1 | cd3÷ldlq fwd s0 / s1 | rev s0 / s1 |", "|---|---|---|---|---|"]
for l in (3, 19, 43):
    row = []; cr = []
    for s in (0, 1):
        o = R.get(("cd", "mul1_i8", s, l, "attn_qg", 2.0)); fw = None
        for f in glob.glob(f"{HERE}/../data/v2/*/cd/s{s}/{l}-attn_qg-K2.jsonl"):
            for x in open(f): fw = json.loads(x)
        row.append(f"{o['rel']['ldlq'] / fw['rel']['ldlq']:.4f}" if o and fw else "–")
        cr.append((f"{fw['rel']['cd3'] / fw['rel']['ldlq']:.4f}" if fw else "–", f"{o['rel']['cd3'] / o['rel']['ldlq']:.4f}" if o else "–"))
    L.append(f"| {l} | {row[0]} | {row[1]} | {cr[0][0]} / {cr[1][0]} | {cr[0][1]} / {cr[1][1]} |")
L.append("")

# ---- item 2 -----------------------------------------------------------------------------------------------------------
I2T = [(10, "ffn_down"), (19, "attn_output"), (19, "attn_qg"), (43, "attn_v"), (30, "ffn_down"), (10, "ffn_gate_up"),
       (10, "ssm_out"), (10, "ssm_qkv"), (50, "ffn_gate_up"), (50, "ssm_qkv")]
ALS = ["1", "1.15", "1.33", "1.5", "1.75", "2"]
cbn = lambda a: "mul1x_i8" if a == "1" else f"mul1x_i8c{a}"
K2S = [6.5, 7.0, 7.25, 7.5, 8.0]
def rel2(cb, s, l, w, K):
    o = R.get(("gain", cb, s, l, w, K)); return o["rel"]["1.0"] if o else None
def ratios(a, K, s):
    out = []
    for l, w in I2T:
        x, y = rel2(cbn(a), s, l, w, K), rel2("mul1x", s, l, w, K)
        if x is None or y is None: return None
        out.append(x / y)
    return np.array(out)
L += ["## Item 2 · #109 clipped i8: r = rel(i8c α) / rel(mul1x), m = 1, after the refit (10 tensors)", "",
      "Median over tensors (sample 0 / sample 1); α = 1 is today's i8.", "",
      "| K | " + " | ".join(f"α {a}" for a in ALS) + " |", "|---|" + "---|" * len(ALS)]
for K in K2S:
    cells = []
    for a in ALS:
        r0, r1 = ratios(a, K, 0), ratios(a, K, 1)
        cells.append(f"{np.median(r0):.3f} / {np.median(r1):.3f}" if r0 is not None and r1 is not None else
                     (f"{np.median(r0):.3f} / –" if r0 is not None else "–"))
    L.append(f"| {K:g} | " + " | ".join(cells) + " |")
L += ["", "α\\*(K) picked on sample 0 (lowest median), judged on sample 1:", "",
      "| K | α* | s1 median | s1 p90 | s1 max | α = 1 s1 median | equal-error mul1 K (median, α*) |", "|---|---|---|---|---|---|---|"]
pick = {}
def eqK(cbname, s, l, w, K):
    """the mul1x K with the same rel, log-linear on mul1x's measured K grid (extrapolated below 6.5 / above 8)"""
    ks = [k for k in K2S if rel2("mul1x", s, l, w, k) is not None]
    if len(ks) < 2: return None
    y = rel2(cbname, s, l, w, K)
    if y is None: return None
    lr = [math.log(rel2("mul1x", s, l, w, k)) for k in ks]; ly = math.log(y)
    for i in range(len(ks) - 1):
        if lr[i] >= ly >= lr[i + 1]: return ks[i] + (ly - lr[i]) / (lr[i + 1] - lr[i]) * (ks[i + 1] - ks[i])
    i = 0 if ly > lr[0] else len(ks) - 2
    return ks[i] + (ly - lr[i]) / (lr[i + 1] - lr[i]) * (ks[i + 1] - ks[i])
for K in K2S:
    m0 = {a: np.median(ratios(a, K, 0)) for a in ALS if ratios(a, K, 0) is not None}
    if len(m0) < len(ALS): L.append(f"| {K:g} | pending | | | | | |"); continue
    a = min(m0, key=m0.get); r1 = ratios(a, K, 1); b1 = ratios("1", K, 1)
    if r1 is None or b1 is None: L.append(f"| {K:g} | {a} | pending | | | | |"); continue
    eq = [eqK(cbn(a), 1, l, w, K) for l, w in I2T]; eq = [e for e in eq if e is not None]
    pick[K] = (a, float(np.median(r1)), float(np.percentile(r1, 90)), float(np.median(b1)))
    L.append(f"| {K:g} | {a} | {np.median(r1):.3f} | {np.percentile(r1, 90):.3f} | {r1.max():.3f} | {np.median(b1):.3f} | {np.median(eq):.2f} |")
v = []
if 7.0 in pick and 6.5 in pick:
    a7, m7, p7, _ = pick[7.0]; a65, m65, _, b65 = pick[6.5]
    closes = m7 <= 1.05 and p7 <= 1.10 and m65 <= b65
    v.append(f"K7 gap: **{'CLOSES' if closes else 'does not close'}** (α* {a7}: median {m7:.3f}, p90 {p7:.3f}; K6.5 α* {a65} {m65:.3f} vs i8 {b65:.3f})")
    for K in (7.25, 7.5):
        if K in pick: v.append(f"i8 K{K:g} {'**joins**' if pick[K][1] <= 1.10 and pick[K][2] <= 1.20 else 'does not join'} (α* {pick[K][0]}: median {pick[K][1]:.3f}, p90 {pick[K][2]:.3f})")
    if not closes and all(pick[K][1] > 1.10 for K in pick if K == 7.0): v.append("→ **negative** by the rule (K7 median > 1.10 at every α)" if min(np.median(ratios(a, 7.0, 1)) for a in ALS) > 1.10 else "")
L += [""] + [f"- {x}" for x in v if x] + [""]

# ---- item 4 -----------------------------------------------------------------------------------------------------------
I4 = {"attn_output": [3, 19, 43, 63], "ssm_out": [2, 10, 30, 50, 62]}
VAR = ["post@K8", "pre@K8", "prepost@K8"]
L += ["## Item 4 · #110 bf16 (row, 128) scales at equal bits: r = rel(variant @ K−⅛) / rel(plain @ K)", "",
      "Per tensor geo over samples 0 / 1; family median (max tensor); `⅛ bit` = rel(plain@K−⅛) / rel(plain@K).", "",
      "| K | family | ⅛ bit | " + " | ".join(VAR) + " |", "|---|---|---|" + "---|" * len(VAR)]
verd = {}
for K in (4.0, 5.0, 6.0, 7.0):
    cb = "mul1x_i8" if K < 6 else "mul1x"
    for fam, ls in I4.items():
        recs = {(l, s): R.get(("rs", cb, s, l, fam, K)) for l in ls for s in (0, 1)}
        if None in recs.values(): L.append(f"| {K:g} | {fam} | pending ({sum(v is not None for v in recs.values())}/{len(recs)}) | | | |"); continue
        eighth = [geo([recs[(l, s)]["rel"]["plain@K8"] / recs[(l, s)]["rel"]["plain@K"] for s in (0, 1)]) for l in ls]
        cells = []
        for var in VAR:
            per = [[recs[(l, s)]["rel"][var] / recs[(l, s)]["rel"]["plain@K"] for s in (0, 1)] for l in ls]
            g = [geo(p) for p in per]; mx = max(max(p) for p in per)
            verd[(K, fam, var)] = (float(np.median(g)), mx)
            cells.append(f"{np.median(g):.3f} ({mx:.3f})")
        L.append(f"| {K:g} | {fam} | {np.median(eighth):.3f} | " + " | ".join(cells) + " |")
L += [""]
for var in VAR:
    ks = [K for K in (5.0, 6.0) if all((K, f, var) in verd for f in I4)]
    if len(ks) < 2: L.append(f"- {var}: pending"); continue
    p = [K for K in ks if all(verd[(K, f, var)][0] <= 0.97 and verd[(K, f, var)][1] <= 1.00 for f in I4)]
    L.append(f"- {var}: **{'worth a real check' if p else 'FAIL'}**" + (f" at K{', K'.join(f'{k:g}' for k in p)}" if p else ""))
# ---- item 5 -----------------------------------------------------------------------------------------------------------
L += ["", "## Item 5 · #106 / #103 on the order-fixed proxy: rel(arm) / rel(natural) (stored-order arms; exllamav3 quantizes",
      "the last stored block first: `head` = heaviest quantized last (literal), `rev` = heaviest first (quantization order))", ""]
def r5(w, l, K, s, arm):
    o = R.get(("order", "mul1_i8", s, l, w, K)); return o["rel"][arm] / o["rel"]["natural"] if o and arm in o["rel"] else None
OLD = {"attn_output": "pass 0.928 / 0.927 / 0.927", "ssm_out": "fail 0.986 / 0.986 / 0.988", "ffn_down": "fail 0.994 / 0.997 / 0.998"}
RULE5 = {"attn_output": ([19, 43, 63], (0, 1)), "ssm_out": ([10, 30, 50], (0, 1)), "ffn_down": ([10, 30], (0,))}
L += ["| family | K | head (literal) | rev (quantization order) | worst tensor rev | rev, samples 0+1 |", "|---|---|---|---|---|---|"]
for fam, (ls, ss) in RULE5.items():
    res = {}
    for arm in ("head", "rev"):
        okK, worst = [], []
        for K in (2.0, 3.0, 4.0):
            v = [r5(fam, l, K, s, arm) for l in ls for s in ss]
            per = [geo([x for x in (r5(fam, l, K, s, arm) for s in ss) if x is not None]) for l in ls]
            res[(arm, K)] = (geo(v) if None not in v else None, max(per) if None not in v else None)
    for K in (2.0, 3.0, 4.0):
        h, r = res[("head", K)], res[("rev", K)]
        both = [r5(fam, l, K, s, "rev") for l in ls for s in (0, 1)]
        L.append(f"| {fam} | {K:g} | {h[0]:.4f} | {r[0]:.4f} | {r[1]:.4f} | {geo([x for x in both if x]) if all(both) else float('nan'):.4f} |"
                 if h[0] and r[0] else f"| {fam} | {K:g} | pending | | | |")
    for arm, name in (("head", "literal stored order"), ("rev", "quantization order")):
        g = [res[(arm, K)][0] for K in (2.0, 3.0, 4.0)]
        if None in g: L.append(f"\n{fam} {name}: pending"); continue
        w = [res[(arm, K)][1] for K in (2.0, 3.0, 4.0)]
        if fam == "ffn_down": ok = sum(x <= 0.98 for x in g) >= 2 and all(x <= 1.00 for x in g)
        else: ok = sum(x <= 0.98 for x in g) >= 2 and all(x <= 1.00 for x in w)
        L.append(f"\n{fam} ({name}): **{'PASS (adopt)' if ok else 'fail'}** — {' / '.join(f'{x:.4f}' for x in g)} (old: {OLD[fam]})")
    ext = [(l, K, r5(fam, l, K, 0, "rev"), r5(fam, l, K, 0, "head")) for l in (0, 43, 63) for K in (2.0, 3.0, 4.0)] if fam == "ffn_down" else []
    if ext: L.append("ffn_down robustness (sample 0, rev / head): " + ", ".join(f"L{l} K{K:g} {a:.4f} / {b:.4f}" for l, K, a, b in ext if a))
# item 5 extension (reported only): every o_proj / out_proj tensor run, sample 0, per K: median [min, max] over tensors
L += ["", "Item 5 extension (reported, not deciding): all run tensors of the family, sample 0, median [min, max] of",
      "rel(arm) / rel(natural); `free` = unconstrained block order (no head grouping), stored order as `head`.", "",
      "| family | K | n | rev | head | free | rev > 1.00 |", "|---|---|---|---|---|---|---|"]
for fam in ("attn_output", "ssm_out"):
    for K in (2.0, 3.0, 4.0):
        ls = sorted({k[3] for k in R if k[0] == "order" and k[4] == fam and k[5] == K and k[2] == 0})
        st = lambda arm: [x for x in (r5(fam, l, K, 0, arm) for l in ls) if x is not None]
        f = lambda v: f"{np.median(v):.4f} [{min(v):.4f}, {max(v):.4f}]" if v else "–"
        rv = st("rev")
        L.append(f"| {fam} | {K:g} | {len(rv)} | {f(rv)} | {f(st('head'))} | {f(st('free'))} | {sum(x > 1 for x in rv)} |")
print("\n".join(L))
