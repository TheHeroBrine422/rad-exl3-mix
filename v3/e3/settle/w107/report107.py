#!/usr/bin/env python3
"""report107.py DATA [--allow-partial] -- w107 #141 (../w107.md) from DATA (= data/w107):
  out/{thor,colo}/   settle107 rows (g1-*, *-K{K}-{cur,tab}.jsonl, *-ntry.jsonl), gates/table.json
  real/              gpubox #141-02: real-{cur,tab}-K{K}.jsonl (real100.py) and log-{arm}-K{K}.jsonl (the jobs' log.jsonl)
  swj/result.txt     gpubox #141-04 pair lines
w100's stored rows (../data/w100/out) are the cur arm at K0.125 / 0.5 / 1; w102's (../data/w102/d3) give T16 and gate G0.
Gates first (G1, G0); a rule is read only on the rows its gate passed."""
import glob, json, math, os, re, sys
from collections import defaultdict

import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); SD = os.path.join(H, "..", "data")
DATA = sys.argv[1]; PARTIAL = "--allow-partial" in sys.argv
geo = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
OLD = [0.125, 0.5, 1.0]; NEW = [0.25, 0.375, 0.625, 0.75, 0.875]; ALL = sorted(OLD + NEW)
INPUTS = {"ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v"}


def rows(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        out += [dict(json.loads(l), _box=f.split("/")[-2]) for l in open(f)]
    return out


def boot_t(d, n=10000, seed=107):
    """d: {tensor: [log ratios of its samples]} -> geo and 95% CI resampling tensors"""
    v = np.array([np.mean(x) for x in d.values()])
    if len(v) < 2: return float(np.exp(v.mean())) if len(v) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed); i = rng.integers(0, len(v), (n, len(v))); m = np.exp(v[i].mean(1))
    return float(np.exp(v.mean())), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def p(*a): print(*a)


# ---------------------------------------------------------------- #141-01
def w100_cur():
    st = {}
    for o in rows(f"{SD}/w100/out/*/*-sub1-K*.jsonl"):
        if o["arm"] == "cur": st[(o["layer"], o["w"], o["sample"], float(o["bits"]))] = o
    return st


def item1():
    p("## #141-01 · LDLQ drift at every sub-1 level (w100's proxy, 27 tensors × 2 samples, i8)\n")
    st = w100_cur()
    g1 = rows(f"{DATA}/out/*/g1-*.jsonl")
    p("**Gate G1** (cur re-run vs w100's stored row; same box = bit-identical expected):\n")
    p("| box | tensor | K | rel2_B rel. diff | d_all rel. diff | w100 box |")
    p("|---|---|---|---|---|---|")
    ok = True
    for o in g1:
        r = st[(o["layer"], o["w"], o["sample"], float(o["bits"]))]
        a = abs(o["rel2"]["B"] / r["rel2"]["B"] - 1); b = abs(o["measured"]["d_all"] / r["measured"]["d_all"] - 1)
        ok &= a <= 1e-6 and b <= 1e-9
        p(f"| {o['_box']} | {o['layer']} {o['w']} s{o['sample']} | {o['bits']:g} | {a:.1e} | {b:.1e} | {r.get('_box')} |")
    p(f"\n**G1: {'PASS' if ok else 'FAIL'}** (rel2_B ≤ 1e-6 and d_all ≤ 1e-9, both relative)\n")
    cur = {k: v for k, v in st.items()}
    for o in rows(f"{DATA}/out/*/*-cur.jsonl"):
        cur[(o["layer"], o["w"], o["sample"], float(o["bits"]))] = o
    tab = {(o["layer"], o["w"], o["sample"], float(o["bits"])): o for o in rows(f"{DATA}/out/*/*-tab.jsonl")}
    T = json.load(open(f"{DATA}/gates/table.json")) if os.path.exists(f"{DATA}/gates/table.json") else {}
    T = {**{0.125: 1.348, 0.5: 1.184, 1.0: 1.089}, **{float(k): v for k, v in T.items()}}
    p("| K | n | D = geo d_all | inputs | rest | min / max | d_fork | T(K) | interp of w100's 3 points P(K) | D / P − 1 |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    D = {}
    for K in ALL:
        c = [o for k, o in cur.items() if k[3] == K]
        if not c: continue
        d = [o["measured"]["d_all"] for o in c]; D[K] = geo(d)
    def interp(K):
        pts = [(k, D[k]) for k in OLD if k in D]
        if len(pts) < 3: return float("nan")
        (k0, d0), (k1, d1) = (pts[0], pts[1]) if K < 0.5 else (pts[1], pts[2])
        return math.exp(math.log(d0) + (math.log(d1) - math.log(d0)) * (K - k0) / (k1 - k0))
    within = []
    for K in ALL:
        c = [o for k, o in cur.items() if k[3] == K]
        if not c: continue
        d = [o["measured"]["d_all"] for o in c]
        di = [o["measured"]["d_all"] for o in c if o["w"] in INPUTS]; dr = [o["measured"]["d_all"] for o in c if o["w"] not in INPUTS]
        P = interp(K) if K in NEW else float("nan")
        if K in NEW and len(c) == 54: within.append(abs(D[K] / P - 1))
        p(f"| {K:g} | {len(c)} | {D[K]:.4f} | {geo(di):.4f} | {geo(dr):.4f} | {min(d):.3f} / {max(d):.3f} | "
          f"{geo([o['measured']['d_fork'] for o in c]):.4f} | {T.get(K, float('nan')):.3f} | "
          f"{'' if K in OLD else f'{P:.4f}'} | {'' if K in OLD else f'{D[K] / P - 1:+.2%}'} |")
    if len(within) == 5:
        three = max(within) <= 0.01
        p(f"\n**Interp check:** max |D / P − 1| = {max(within):.2%} → "
          f"{'**three points suffice** (log-linear interp between 1.348 / 1.184 / 1.089)' if three else '**the eight-point table is needed**'}\n")
    else:
        three = None; p(f"\nInterp check: {len(within)} of 5 new K complete\n")
    p("Drift arm (search with T(K), quantized only if the scale moves) vs cur, held-out two-sided rel2_B, paired; "
      "tensor-bootstrap 95% CI (10,000 draws, tensors resampled with both samples):\n")
    p("| K | T(K) | n | moved | g cur → tab (median) | r = geo rel2_B tab / cur | 95% CI | inputs | rest | rel1_pub ratio | tensor×samples better | proposed (r ≤ 1.000) |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    prop = {}
    for K in ALL:
        pr = [(cur[k], o) for k, o in tab.items() if k[3] == K and k in cur]
        if not pr: continue
        lr = defaultdict(list); gi, gr, r1, better, mv, gc, gt = [], [], [], 0, 0, [], []
        for c, t in pr:
            if t.get("same"): x = 1.0; y = 1.0
            else: x = t["rel2"]["B"] / c["rel2"]["B"]; y = t["rel1"]["pub"] / c["rel1"]["pub"]; mv += 1
            lr[(c["layer"], c["w"])].append(math.log(x)); r1.append(y); better += x < 1
            (gi if c["w"] in INPUTS else gr).append(x)
            gc.append(c["search"]["g_scale"] if c.get("search") else c["s"] / t["s"] * t["search"]["g_scale"])
            gt.append(t["search"]["g_scale"])
        g, lo, hi = boot_t(lr)
        full = len(pr) == 54; prop[K] = (g <= 1.0) if full else None
        p(f"| {K:g} | {T.get(K, float('nan')):.3f} | {len(pr)} | {mv} | {np.median(gc):.3f} → {np.median(gt):.3f} | **{g:.4f}** | "
          f"[{lo:.4f}, {hi:.4f}] | {geo(gi):.4f} | {geo(gr):.4f} | {geo(r1):.4f} | {better}/{len(pr)} | "
          f"{'—' if not full else ('**yes**' if g <= 1.0 else 'no (1.08 stays)')} |")
    if all(prop.get(K) is not None for K in ALL):
        pts = [K for K in ALL if prop[K]]
        use = [K for K in (OLD if three else ALL) if prop[K]]
        p(f"\n**Proposal (#141-01):** T(K) at {', '.join(f'K{K:g} {T[K]:.3f}' for K in use) or 'no K'} "
          f"({'three-point table, ourq / rxq log-linear interp between' if three else 'eight-point table'}); "
          f"K ≥ 1.5 unchanged. Points with r > 1.000: {', '.join(f'K{K:g}' for K in ALL if not prop[K]) or 'none'}.\n")
    return D


# ---------------------------------------------------------------- #141-02
def item2():
    p("## #141-02 · g and error on the production quantizer (rxq, layers 2–3, 13 tensors, i8, seed 0)\n")
    any_ = False
    res = {}
    for K in ("0.125", "0.5"):
        R = {}
        for arm in ("cur", "tab"):
            f = f"{DATA}/real/real-{arm}-K{K}.jsonl"; lg = f"{DATA}/real/log-{arm}-K{K}.jsonl"
            if not os.path.exists(f): continue
            g = {}
            if os.path.exists(lg):
                for l in open(lg):
                    o = json.loads(l)
                    if "key" in o: g[o["key"]] = (o.get("g_scale"), o.get("drift"))
            R[arm] = {json.loads(l)["key"]: dict(json.loads(l), g=g.get(json.loads(l)["key"], (None, None))) for l in open(f)}
        if len(R) < 2: p(f"K{K}: arms present {list(R)}\n"); continue
        any_ = True
        keys = sorted(set(R["cur"]) & set(R["tab"]), key=lambda k: (int(k.split(".")[3]), k))
        p(f"### K{K}\n")
        p("| tensor | g cur (drift) | g tab (drift) | rel2 cur | rel2 tab | r2 = tab / cur | rel1 cur | rel1 tab | r1 |")
        p("|---|---|---|---|---|---|---|---|---|")
        r2s, r1s = [], []
        for k in keys:
            c, t = R["cur"][k], R["tab"][k]; r2 = t["rel2"] / c["rel2"]; r1 = t["rel1"] / c["rel1"]; r2s.append(r2); r1s.append(r1)
            gc, dc = c["g"]; gt, dt = t["g"]
            fmt = lambda g, d: "?" if g is None else f"{g:.3f}{' (edge)' if g >= 2.05 - 1e-6 else ''} ({d:.3f})"
            p(f"| {k.replace('model.language_model.layers.', 'L')} | {fmt(gc, dc)} | {fmt(gt, dt)} | {c['rel2']:.4f} | {t['rel2']:.4f} | "
              f"{r2:.4f} | {c['rel1']:.4f} | {t['rel1']:.4f} | {r1:.4f} |")
        nb = sum(r <= 1 for r in r2s); worst = max(r2s)
        edge = {a: sum(1 for k in keys if R[a][k]["g"][0] is not None and R[a][k]["g"][0] >= 2.05 - 1e-6) for a in R}
        p(f"\nK{K}: geo r2 {geo(r2s):.4f}, geo r1 {geo(r1s):.4f}; r2 ≤ 1 on {nb}/{len(keys)}, worst r2 {worst:.4f}; "
          f"r1 ≤ 1 on {sum(r <= 1 for r in r1s)}/{len(keys)}; at the g edge: cur {edge.get('cur')}, tab {edge.get('tab')}\n")
        res[K] = (nb, len(keys), worst)
    if len(res) == 2:
        ok = all(n >= 11 and m == 13 and w <= 1.005 for n, m, w in res.values())
        p(f"**#141-02 rule (r2 ≤ 1 on ≥ 11/13 at both K, none > 1.005): {'PASS → the drift table is proposed for adoption' if ok else 'FAIL → not proposed from this check'}**\n")
    elif not any_:
        p("(no data yet)\n")


# ---------------------------------------------------------------- #141-03
def item3():
    p("## #141-03 · NTRY 4 at the sub-1 levels (w102's -01 setup, 27 tensors × s0, i8)\n")
    st = {}
    for f in glob.glob(f"{SD}/w102/d3/*.jsonl"):
        for l in open(f):
            o = json.loads(l); st[(o["layer"], o["w"], float(o["bits"]), o["arm"])] = o
    by = defaultdict(dict)
    for o in rows(f"{DATA}/out/*/*-ntry.jsonl"):
        by[(o["layer"], o["w"], float(o["bits"]))][o["arm"]] = o
    worst, n0, bad = 0.0, 0, []
    for k, a in by.items():
        if "T1" not in a: continue
        r = st.get((*k, "T1"))
        if r is None: bad.append(k); continue
        d = abs(a["T1"]["rel"]["B"] / r["rel"]["B"] - 1); worst = max(worst, d); n0 += 1
        if d > 1e-6: bad.append((k, d))
    g0 = not bad and n0 > 0
    p(f"**Gate G0** (T1 = w102's stored T1 row, 1e-6 relative): {n0} rows, worst {worst:.1e} → **{"PASS" if g0 else ("no rows yet" if n0 == 0 and not bad else "FAIL")}** {bad[:5] if bad else ""}\n")
    if not g0: return
    p("| K | n | R4 = geo T4 / T1 | 95% CI | inputs | rest | R16 (w102) | share of NTRY 16's gain | T16 / T4 | R4 ≤ 0.995 | CPU factor sec T4 / T1 (median) |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    fac = {}
    for K in OLD:
        ks = [k for k in by if k[2] == K and "T4" in by[k]]
        if not ks: continue
        lr = {k: [math.log(by[k]["T4"]["rel"]["B"] / by[k]["T1"]["rel"]["B"])] for k in ks}
        g, lo, hi = boot_t(lr)
        gi = [math.exp(v[0]) for k, v in lr.items() if k[1] in INPUTS]; gr = [math.exp(v[0]) for k, v in lr.items() if k[1] not in INPUTS]
        r16 = geo([st[(*k, "T16")]["rel"]["B"] / st[(*k, "T1")]["rel"]["B"] for k in ks])
        t164 = geo([st[(*k, "T16")]["rel"]["B"] / by[k]["T4"]["rel"]["B"] for k in ks])
        f = float(np.median([by[k]["T4"]["sec"] / by[k]["T1"]["sec"] for k in ks])); fac[K] = f
        p(f"| {K:g} | {len(ks)} | **{g:.4f}** | [{lo:.4f}, {hi:.4f}] | {geo(gi):.4f} | {geo(gr):.4f} | {r16:.4f} | "
          f"{(1 - g) / (1 - r16):.0%} | {t164:.4f} | {'yes' if g <= 0.995 else 'no'} | {f:.2f} |")
    p("\nCard-0 level time (w102's tile-search shares s; factor (1 − s) + s · f):\n")
    p("| K | today | s | f = 4 (GPU passes ∝ NTRY) | f = CPU factor |")
    p("|---|---|---|---|---|")
    for K, s, h in ((0.125, 0.89, 3.3), (1.0, 0.67, 1.0)):
        if K not in fac: continue
        a = (1 - s) + s * 4; b = (1 - s) + s * fac[K]
        p(f"| {K:g} | {h:.1f} h | {s:.2f} | ×{a:.2f} → {h * a:.1f} h | ×{b:.2f} → {h * b:.1f} h |")
    p("")


# ---------------------------------------------------------------- #141-04
def item4():
    p("## #141-04 · swj at sub-1 (#132-01; 0.8B, today's rxq stack, g08-search1)\n")
    f = f"{DATA}/swj/result.txt"
    if not os.path.exists(f): p("(no data yet)\n"); return
    pr = {}
    for l in open(f):
        m = re.search(r"pair K([0-9.]+)s(\d) base -> swj: .*\"ratio\": ([0-9.]+), \"diff\": ([0-9.e-]+), \"se\": ([0-9.e-]+), \"z\": ([0-9.e-]+)", l)
        if m: pr[(float(m.group(1)), int(m.group(2)))] = (float(m.group(3)), float(m.group(6)))
    p("| K | seed 0 ratio (z) | seed 1 ratio (z) | r = geo | swj keeps K (r ≤ 1.000, no seed > 1.005) |")
    p("|---|---|---|---|---|")
    for K in (0.5, 1.0):
        a, b = pr.get((K, 0)), pr.get((K, 1))
        if not (a and b): p(f"| {K:g} | {a} | {b} | | incomplete |"); continue
        r = math.sqrt(a[0] * b[0]); keep = r <= 1.0 and max(a[0], b[0]) <= 1.005
        p(f"| {K:g} | {a[0]:.4f} ({a[1]:+.1f}) | {b[0]:.4f} ({b[1]:+.1f}) | **{r:.4f}** | {'**keeps**' if keep else '**no → propose swj k_min above K' + format(K, 'g') + '**'} |")
    p("")


if __name__ == "__main__":
    p("# w107 report (#141)\n")
    item1(); item2(); item3(); item4()
