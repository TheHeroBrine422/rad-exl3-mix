#!/usr/bin/env python3
"""agg.py RESULTS_DIR EXPERTS.json [SLICES=0] -- w211 report tables (markdown on stdout) from the res-*.jsonl files.

Per (matrix, K): baseline = best own-H relative error over factors {1, 1.15, 1.3} of setting 'base'; wide = best over all six;
every other setting = best over its factors {1, 1.15, 1.3}; ratio = setting / baseline (<1 = better). Slices (disjoint row samples) are
averaged per (matrix, setting, factor) before the best factor is taken (SLICES=0,1 -> both). Classes (hot / medium / cold x
gate_up / down) have 6 matrices each; CI = 95 % percentile bootstrap over matrices of the geo-mean ratio (all 18 per kind too).
rel_own is measured under the expert's own H (in-sample for the own-H settings); rel_pool under the pooled ffn_in H (gate_up only).
"""
import glob, json, math, os, sys
import numpy as np

d, ejson = sys.argv[1], sys.argv[2]
slices = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0").split(",")]
ex = json.load(open(ejson))
cls_of = {}; info = {}
for L, v in ex.items():
    for c in ("hot", "medium", "cold"):
        for x in v[c]:
            cls_of[(int(L), x["e"])] = c; info[(int(L), x["e"])] = x
R = {}      # (layer,e,kind,setting,K,slice) -> allf
META = {}
for fn in glob.glob(os.path.join(d, "res-*.jsonl")):
    for l in open(fn):
        r = json.loads(l)
        if "meta" in r: META[(r["layer"], r["e"], r["kind"])] = r["meta"]; continue
        R[(r["layer"], r["e"], r["kind"], r["setting"], r["K"], r["slice"])] = r["allf"]
F3 = ("1.0", "1.15", "1.3")
rng = np.random.default_rng(0)


def comb(layer, e, kind, s, K, metric):
    """factor -> mean over the available requested slices of metric (None if any slice missing)"""
    arrs = [R.get((layer, e, kind, s, K, sl)) for sl in slices]
    if any(a is None for a in arrs): return None
    out = {}
    for f in arrs[0]:
        if all(f in a and metric in a[f] for a in arrs): out[f] = float(np.mean([a[f][metric] for a in arrs]))
    return out or None


def best(layer, e, kind, s, K, metric="own", fset=F3):
    c = comb(layer, e, kind, s, K, metric)
    if c is None: return None
    c = {f: v for f, v in c.items() if f in fset}
    if not c: return None
    f = min(c, key=c.get); return c[f], f


def gm_ci(x):
    x = np.log(np.asarray(x)); g = x.mean()
    bs = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return math.exp(g), math.exp(np.percentile(bs, 2.5)), math.exp(np.percentile(bs, 97.5))


mats = sorted({(k[0], k[1], k[2]) for k in R})
Ks = [1, 2, 3, 4]
settings = ["d0.001", "d0.005", "d0.025", "d0.05", "d0.1", "pooled", "mix0.1", "mix1"]
print(f"slices {slices}; matrices with results: {len(mats)}\n")
groups = [(c, k) for k in ("gate_up", "down") for c in ("hot", "medium", "cold")] + [("all", "gate_up"), ("all", "down")]


def members(c, k):
    return [m for m in mats if m[2] == k and (c == "all" or cls_of[(m[0], m[1])] == c)]


print("## 1. Baseline relative error (own H, factors {1,1.15,1.3}, damp 0.01): median over matrices [min .. max]\n")
print("| class | kind | n | " + " | ".join(f"K{K}" for K in Ks) + " |\n|---|---|---|" + "---|" * len(Ks))
for c, k in groups:
    row = []
    for K in Ks:
        v = [best(l, e, kk, "base", K)[0] for (l, e, kk) in members(c, k) if best(l, e, kk, "base", K)]
        row.append(f"{np.median(v):.4g} [{min(v):.3g}..{max(v):.3g}]" if v else "-")
    print(f"| {c} | {k} | {len(members(c, k))} | " + " | ".join(row) + " |")

print("\n## 2. Per-matrix baseline rel (own H) and chosen factors (baseline subset / wide grid)\n")
print("| matrix | class | kept rows | " + " | ".join(f"K{K} rel (f3 / f wide)" for K in Ks) + " |\n|---|---|---|" + "---|" * len(Ks))
for (l, e, k) in mats:
    row = []
    for K in Ks:
        b = best(l, e, k, "base", K); w = best(l, e, k, "base", K, fset=("0.8", "0.9", "1.0", "1.15", "1.3", "1.5"))
        row.append(f"{b[0]:.4g} ({b[1]} / {w[1]})" if b else "-")
    print(f"| L{l} e{e} {k} | {cls_of[(l, e)]} | {info[(l, e)]['kept_rows']} | " + " | ".join(row) + " |")

print("\n## 3. Ratio of each setting to the baseline (rel_own setting / rel_own baseline; geo-mean [95 % bootstrap CI over matrices]; <1 better)\n")
for k in ("gate_up", "down"):
    print(f"\n### {k}\n")
    print("| class | n | K | wide factors | " + " | ".join(settings) + " |\n|---|---|---|---|" + "---|" * len(settings))
    for c in ("hot", "medium", "cold", "all"):
        for K in Ks:
            row = []
            ms = members(c, k)
            wr = [best(l, e, kk, "base", K, fset=("0.8", "0.9", "1.0", "1.15", "1.3", "1.5"))[0] / best(l, e, kk, "base", K)[0]
                  for (l, e, kk) in ms if best(l, e, kk, "base", K)]
            cells = [f"{gm_ci(wr)[0]:.4f} [{gm_ci(wr)[1]:.4f}, {gm_ci(wr)[2]:.4f}]" if wr else "-"]
            for s in settings:
                rr = []
                for (l, e, kk) in ms:
                    a, b = best(l, e, kk, s, K), best(l, e, kk, "base", K)
                    if a and b: rr.append(a[0] / b[0])
                cells.append(f"{gm_ci(rr)[0]:.4f} [{gm_ci(rr)[1]:.4f}, {gm_ci(rr)[2]:.4f}]" if len(rr) >= 2 else "-")
            print(f"| {c} | {len(ms)} | {K} | " + " | ".join(cells) + " |")

print("\n## 4. Pooled / mixed Hessian judged under the pooled H too (gate_up; rel_pool ratio to baseline's rel_pool, all 18 matrices)\n")
print("| K | " + " | ".join(["pooled", "mix0.1", "mix1", "d0.1"]) + " |\n|---|---|---|---|---|")
for K in Ks:
    cells = []
    for s in ["pooled", "mix0.1", "mix1", "d0.1"]:
        rr = []
        for (l, e, kk) in members("all", "gate_up"):
            a, b = best(l, e, kk, s, K, "pool"), best(l, e, kk, "base", K, "pool")
            if a and b: rr.append(a[0] / b[0])
        cells.append(f"{gm_ci(rr)[0]:.4f} [{gm_ci(rr)[1]:.4f}, {gm_ci(rr)[2]:.4f}]" if len(rr) >= 2 else "-")
    print(f"| {K} | " + " | ".join(cells) + " |")

print("\n## 5. Chosen scale factor of each setting (counts over matrices, best of its factors)\n")
for k in ("gate_up", "down"):
    print(f"\n### {k}\n\n| setting | " + " | ".join(f"K{K}" for K in Ks) + " |\n|---|---|---|---|---|")
    for s in ["base"] + settings:
        if k == "down" and s in ("pooled", "mix0.1", "mix1"): continue
        row = []
        for K in Ks:
            cnt = {}
            for (l, e, kk) in members("all", k):
                b = best(l, e, kk, s, K, fset=("0.8", "0.9", "1.0", "1.15", "1.3", "1.5") if s == "base" else F3)
                if b: cnt[b[1]] = cnt.get(b[1], 0) + 1
            row.append(" ".join(f"{f}:{n}" for f, n in sorted(cnt.items(), key=lambda t: float(t[0]))))
        print(f"| {s} | " + " | ".join(row) + " |")

print("\n## 6. Kept rows vs the per-expert-H effect (gate_up: baseline vs pooled; Spearman over matrices)\n")
from itertools import combinations
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])
print("| kind | K | corr(kept rows, pooled/base ratio) | corr(log count, pooled/base ratio) | n | ratio range |\n|---|---|---|---|---|---|")
for k in ("gate_up",):
    for K in Ks:
        xs, cs, ys = [], [], []
        for (l, e, kk) in members("all", k):
            a, b = best(l, e, kk, "pooled", K), best(l, e, kk, "base", K)
            if a and b: xs.append(info[(l, e)]["kept_rows"]); cs.append(math.log(info[(l, e)]["count"])); ys.append(a[0] / b[0])
        if len(ys) > 3: print(f"| {k} | {K} | {spearman(xs, ys):+.2f} | {spearman(cs, ys):+.2f} | {len(ys)} | {min(ys):.3f}..{max(ys):.3f} |")

print("\n## 7. MXFP4 grid structure after rotation (all rows; excess kurtosis, Gaussian = 0; KS = max |ECDF - N(0,1) CDF| on <= 2M values)\n")
print("| matrix | class | kurt unrotated (unit-RMS rows) | kurt rotated | kurt rotated, per-row mean | gaussian ref kurt (same size) | KS rotated | h eff.rank (tr^2/||H||F^2) | hdiag dead |\n|---|---|---|---|---|---|---|---|---|")
for (l, e, k) in mats:
    m = META.get((l, e, k))
    if m: print(f"| L{l} e{e} {k} | {cls_of[(l, e)]} | {m['kurt_unrot']:+.4f} | {m['kurt_rot']:+.4f} | {m['kurt_rot_row_mean']:+.4f} | {m['kurt_gauss_ref']:+.4f} | {m['ks_rot']:.4f} | {m['h_eff_rank_ratio']:.1f} | {m['hdiag_dead']} |")
