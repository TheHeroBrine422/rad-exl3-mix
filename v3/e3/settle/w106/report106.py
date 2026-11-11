#!/usr/bin/env python3
"""report106.py A -- #140 (../w106.md) readout: KL of every (K, seed, arm) on g08-search1 (an arm whose c equals an
earlier arm's on every writer reuses that arm's score: TAG-<arm>.same), ratios vs cur with a per-document bootstrap
(pcmp.py's arithmetic: documents resampled, each its gen rows' KL sum and count; 2,000 resamples, seed 0) 95% interval
and z, the pass rule, and the C9 statistics (TAG-c9.json)."""
import json, math, os, sys
import numpy as np

A = sys.argv[1]
G = json.load(open(f"{A}/g08-search1/kld.json"))
KS, SEEDS, ARMS = (0.5, 1, 1.5, 2), (0, 1), ("off", "cur", "ak", "e3")
ks = lambda K: f"{K:g}"
rng = np.random.default_rng(0); ND = len(G["docs"]); IDX = rng.integers(0, ND, (2000, ND))


def docsums(f):
    X = np.fromfile(f, np.float32).reshape(-1, 4); assert len(X) == G["rows"], f
    s, n, t = [], [], []
    for d in G["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); g = X[sl, 2] > 0
        s.append(X[sl, 0][g].sum()); n.append(g.sum()); t.append(X[sl, 1][g].sum())
    return np.array(s, np.float64), np.array(n, np.float64), np.array(t, np.float64)


def resolve(tag, arm):
    seen = set()
    while os.path.exists(f"{A}/mix/{tag}-{arm}.same") and arm not in seen:
        seen.add(arm); arm = open(f"{A}/mix/{tag}-{arm}.same").read().strip()
    return arm


S = {}
for K in KS:
    for s in SEEDS:
        tag = f"K{ks(K)}s{s}"
        for arm in ARMS:
            src = resolve(tag, arm); f = f"{A}/ev/{tag}-{src}.f32"
            if os.path.exists(f): S[(K, s, arm)] = (docsums(f), src)


def pair(a, b):
    (sa, n, _), (sb, _, _) = a, b
    r = sb.sum() / sa.sum(); rb = sb[IDX].sum(1) / sa[IDX].sum(1)
    d = (sb.sum() - sa.sum()) / n.sum(); se = ((sb[IDX].sum(1) - sa[IDX].sum(1)) / n[IDX].sum(1)).std()
    return r, np.quantile(rb, 0.025), np.quantile(rb, 0.975), d / se if se > 0 else 0.0


out = ["# w106 · #140 report (C9 at low K, 0.8B, g08-search1)", ""]
out += ["## KL per cell (gen-masked mean, top-1)", "", "| K | seed | " + " | ".join(ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
for K in KS:
    for s in SEEDS:
        row = []
        for arm in ARMS:
            if (K, s, arm) not in S: row.append("—"); continue
            (sm, n, t), src = S[(K, s, arm)]
            row.append(f"{sm.sum() / n.sum():.5f} / {t.sum() / n.sum():.4f}" + ("" if src == arm else f" (= {src})"))
        out.append(f"| {ks(K)} | {s} | " + " | ".join(row) + " |")
R = {}
out += ["", "## Ratios vs cur (arm / cur; 95% per-document bootstrap interval; z of the paired difference)", "",
        "| K | seed | off / cur | ak / cur | e3 / cur |", "|---|---|---|---|---|"]
for K in KS:
    for s in SEEDS:
        row = []
        for arm in ("off", "ak", "e3"):
            if (K, s, arm) in S and (K, s, "cur") in S:
                if S[(K, s, arm)][1] == S[(K, s, "cur")][1]: R[(K, s, arm)] = 1.0; row.append("1 (identical)"); continue
                r, lo, hi, z = pair(S[(K, s, "cur")][0], S[(K, s, arm)][0]); R[(K, s, arm)] = r
                row.append(f"{r:.4f} [{lo:.4f}, {hi:.4f}] z {z:+.1f}")
            else: row.append("—")
        out.append(f"| {ks(K)} | {s} | " + " | ".join(row) + " |")
out += ["", "## Pass rule (r ≤ 0.99 at K0.5 and K1 on both seeds; K2 r ≤ 1.003 on both seeds)", ""]
need = [(K, s) for K in (0.5, 1, 2) for s in SEEDS]
passing = []
for arm in ("off", "ak", "e3"):
    if not all((K, s, arm) in R for K, s in need): out.append(f"- {arm}: incomplete"); continue
    lowk = [R[(K, s, arm)] for K in (0.5, 1) for s in SEEDS]; k2 = [R[(2, s, arm)] for s in SEEDS]
    ok = all(r <= 0.99 for r in lowk) and all(r <= 1.003 for r in k2)
    geo = math.exp(sum(map(math.log, lowk)) / 4); geo2 = math.exp(sum(map(math.log, k2)) / 2)
    out.append(f"- **{arm}: {'PASS' if ok else 'fail'}** · K0.5 / K1 per seed {', '.join(f'{r:.4f}' for r in lowk)} (geo {geo:.4f}) · "
               f"K2 {', '.join(f'{r:.4f}' for r in k2)} (geo {geo2:.4f})")
    if ok: passing.append((geo, arm))
out.append("")
out.append(f"**Verdict:** {'propose ' + min(passing)[1] + ' (lowest geo over K0.5 / K1 × seeds among the passing arms)' if passing else 'no arm passes: C9 stays as today'}")
out += ["", "## C9 statistics per K and family (rows pooled over the family's writers; per seed)", "",
        "| K | seed | arm | family (α, e_clip) | c_mean | c_q50 | c_max | c_min | rows at clip | e_mean | e_max |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for K in KS:
    for s in SEEDS:
        f = f"{A}/mix/K{ks(K)}s{s}-c9.json"
        if not os.path.exists(f): continue
        c = json.load(open(f))
        for arm in ARMS:
            for fam, v in c["arms"][arm].items():
                out.append(f"| {ks(K)} | {s} | {arm} | {fam} ({v['alpha']:g}, {v['e_clip']:g}) | {v['c_mean']:.4f} | {v['c_q50']:.4f} | "
                           f"{v['c_max']:.4f} | {v['c_min']:.4f} | {100 * v['at_clip']:.1f}% | {v['e_mean']:.4f} | {v['e_max']:.4f} |")
        out.append(f"| {ks(K)} | {s} | G1 | cur c vs the jobs' sidecars: max rel {c['g1_max_rel']:.2e} ({'PASS' if c['g1_pass'] else 'FAIL'}) | | | | | | | |")
print("\n".join(out))
