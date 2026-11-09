#!/usr/bin/env python3
"""report94.py DATA OUT.md -- the readouts and verdicts of RULES.md (#124, #125) from:
  DATA/sim-{search,qad}.npz (+ -plans.json)   headsim94 (exact head-only KL per row, tau grid, lse, argmax rank)
  DATA/meta/{search,qad}/{doc.i32,gen.u8}     headx94's row bookkeeping;  DATA/gate-{search,qad}.json  G1
  DATA/kl/{search,qad}/<tag>.npz             hq94 score (kA, kB, kC, wB, wC) per proxy arm (if present)
Row sets: search-all, search-gen, qad-gen; A6's halves qad-odd / qad-even = gen rows of odd / even calib documents.
Means are over rows (as headsim); intervals: 2,000 document-bootstrap draws of the ratio of row sums (paired)."""
import json, math, os, re, sys
import numpy as np

D, OUT = sys.argv[1:3]
GUARD = 20000
rng = np.random.default_rng(0)
L = []
def p(s=""): L.append(s); print(s)


def load(s):
    z = np.load(f"{D}/sim-{s}.npz"); m = {k: z[k] for k in z.files}
    m["doc"] = np.fromfile(f"{D}/meta/{s}/doc.i32", np.int32); m["gen"] = np.fromfile(f"{D}/meta/{s}/gen.u8", np.uint8).astype(bool)
    assert len(m["doc"]) == m["kl"].shape[1]
    m["ix"] = {str(n): i for i, n in enumerate(m["names"])}
    return m


S = {s: load(s) for s in ("search", "qad")}
SETS = {"search-all": ("search", np.ones_like(S["search"]["gen"])), "search-gen": ("search", S["search"]["gen"]),
        "qad-gen": ("qad", S["qad"]["gen"]), "qad-odd": ("qad", S["qad"]["gen"] & (S["qad"]["doc"] % 2 == 1)),
        "qad-even": ("qad", S["qad"]["gen"] & (S["qad"]["doc"] % 2 == 0))}


def docsum(rs, v):
    s, msk = SETS[rs]; d = S[s]["doc"][msk]; u, inv = np.unique(d, return_inverse=True)
    return np.bincount(inv, v[msk].astype(np.float64)), np.bincount(inv), len(u)


def mean(rs, v): s, msk = SETS[rs]; return float(v[msk].mean())


def ratio(rs, a, b, draws=2000):
    """sum a / sum b over the row set, with a document-bootstrap 95% interval"""
    sa, _, nd = docsum(rs, a); sb, _, _ = docsum(rs, b)
    bi = rng.integers(0, nd, (draws, nd)); r = sa[bi].sum(1) / sb[bi].sum(1)
    return float(sa.sum() / sb.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def kl(s, n): return S[s]["kl"][S[s]["ix"][n]]
def fmt(r): return f"{r[0]:.4f} [{r[1]:.4f}, {r[2]:.4f}]"


res = {"gates": {s: json.load(open(f"{D}/gate-{s}.json")) for s in ("search", "qad")}}
p("# w94 report (#124 A2b, #125 A6)\n")
p("## G1 (dumps = the golden's rows)\n")
for s, g in res["gates"].items():
    p(f"- {s}: gen rows {g['gen_checked']}, KL vs top-512 {g['kl_top_gen']:.2e} (p99 {g['kl_top_p99_gen']:.2e}), top-1 "
      f"{g['top1_gen']:.4f}; all rows {g['kl_top_all']:.2e} / {g['top1_all']:.4f} -> {'PASS' if g['pass'] else 'FAIL'}")
if not all(g["pass"] for g in res["gates"].values()): p("\n**G1 FAIL: nothing below is a result.**")

# ---------------------------------------------------------------- #124
p("\n## #124 head-only KL on gen rows vs prefill-like rows\n")
RS = ("search-all", "search-gen", "qad-gen")
for rs in RS:
    s, msk = SETS[rs]; p(f"- {rs}: {int(msk.sum())} rows, {len(np.unique(S[s]['doc'][msk]))} docs, guard rows "
                         f"{int((S[s]['top'][msk] >= GUARD).sum())}")
main = [n for n in ("u3", "u4", "u5", "u6", "u7", "s4-3", "s5-3", "s6-3") if n in S["search"]["ix"]]
p("\n| plan | bits | search-all | search-gen | qad-gen | gen / all | qad / search (gen) | guard KL search-gen | top-1 search-gen |")
p("|---|---|---|---|---|---|---|---|---|")
res["plans"] = {}
for n in main:
    a, g, q = mean("search-all", kl("search", n)), mean("search-gen", kl("search", n)), mean("qad-gen", kl("qad", n))
    s, msk = SETS["search-gen"]; gm = msk & (S["search"]["top"] >= GUARD)
    gk = float(kl("search", n)[gm].mean()); t1 = float(S["search"]["hit"][S["search"]["ix"][n]][msk].mean())
    bits = float(S["search"]["avg_bits"][S["search"]["ix"][n]])
    res["plans"][n] = {"bits": bits, "search_all": a, "search_gen": g, "qad_gen": q, "guard_search_gen": gk, "top1_search_gen": t1}
    p(f"| {n} | {bits:.3f} | {a:.6f} | {g:.6f} | {q:.6f} | {g / a:.3f} | {q / g:.3f} | {gk:.6f} | {t1:.4f} |")
p("\nPredicted gains (paired ratio, 95% document-bootstrap CI):\n")
p("| pair | search-all | search-gen | qad-gen |"); p("|---|---|---|---|")
res["gains"] = {}
for a, b in (("s4-3", "u4"), ("s5-3", "u5"), ("s6-3", "u6")):
    if a not in S["search"]["ix"]: continue
    r = {rs: ratio(rs, kl(SETS[rs][0], a), kl(SETS[rs][0], b)) for rs in RS}; res["gains"][f"{a}/{b}"] = r
    p(f"| {a} / {b} | " + " | ".join(fmt(r[rs]) for rs in RS) + " |")

# curve: winner (floor) per budget, hull
p("\n### Curve (today's K3-K7 donors): best plan per budget\n")
cur = {}
for n in S["search"]["ix"]:
    mm = re.match(r"s([\d.]+)-f([\d.]+)$", n)
    if mm: cur.setdefault(float(mm.group(1)), []).append(n)
p("| budget | search-all | search-gen | qad-gen | slab/uniform search-all | search-gen | qad-gen | winners differ: paired CI on search-gen |")
p("|---|---|---|---|---|---|---|---|")
res["curve"] = {}; changes = []
for B in sorted(cur):
    ns = cur[B]; win = {rs: min(ns, key=lambda n: mean(rs, kl(SETS[rs][0], n))) for rs in RS}
    lo = max(k for k in (3, 4, 5, 6, 7) if k <= B + 1e-9); u = f"u{lo}"
    su = {rs: mean(rs, kl(SETS[rs][0], win[rs])) / mean(rs, kl(SETS[rs][0], u)) for rs in RS}
    note = ""
    for o in ("search-all", "qad-gen"):
        if win[o] != win["search-gen"]:
            r = ratio("search-gen", kl("search", win[o]), kl("search", win["search-gen"]))
            sig = r[1] > 1 or r[2] < 1
            note += f"{o} {win[o]} vs {win['search-gen']}: {fmt(r)}{' **' if sig else ''}; "
            if sig: changes.append((B, o, win[o], win["search-gen"], r))
    res["curve"][B] = {"winner": win, "slab_over_uniform": su, "uniform": u}
    p(f"| {B:g} | {win['search-all']} | {win['search-gen']} | {win['qad-gen']} | {su['search-all']:.3f} | "
      f"{su['search-gen']:.3f} | {su['qad-gen']:.3f} | {note} |")
res["curve_changes"] = [list(map(str, c[:4])) + [list(c[4])] for c in changes]
p(f"\nWinners that change with a paired 95% CI excluding 1 on search-gen: {len(changes)}")


def hull(rs):
    pts = sorted((float(S[SETS[rs][0]]["avg_bits"][i]), mean(rs, S[SETS[rs][0]]["kl"][i]), str(n))
                 for n, i in S[SETS[rs][0]]["ix"].items())
    h = []
    for x, y, n in pts:                     # lower convex hull in (bits, log KL)
        ly = math.log(y)
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (x - h[-2][0]) >= (ly - h[-2][1]) * (h[-1][0] - h[-2][0]): h.pop()
        if h and abs(x - h[-1][0]) < 1e-9:
            if ly < h[-1][1]: h[-1] = (x, ly, n)
            continue
        h.append((x, ly, n))
    return [n for _, _, n in h]
res["hull"] = {rs: hull(rs) for rs in RS}
for rs in RS: p(f"- hull {rs}: {' '.join(res['hull'][rs])}")

# tau
p("\n### Logit temperature (exact, real donors)\n")
res["tau"] = {}
taus = [float(t) for t in S["search"]["taus"]]; tps = [str(n) for n in S["search"]["tau_plans"]]
p("| plan | set | " + " | ".join(f"τ {t:g}" for t in taus) + " |"); p("|---|---|" + "---|" * len(taus))
for i, n in enumerate(tps):
    for rs in ("search-gen", "qad-odd", "qad-even"):
        s = SETS[rs][0]; base = mean(rs, kl(s, n))
        rr = [mean(rs, S[s]["kl_tau"][i, j]) / base for j in range(len(taus))]
        res["tau"].setdefault(n, {})[rs] = rr
        p(f"| {n} | {rs} | " + " | ".join(f"{x:.4f}" for x in rr) + " |")
tv = {}
for n in ("u3", "u4"):
    if n not in res["tau"]: continue
    j = int(np.argmin(res["tau"][n]["qad-odd"])); t = taus[j]
    ok = min(res["tau"][n]["qad-odd"]) < 1 and res["tau"][n]["search-gen"][j] <= 0.97 and res["tau"][n]["qad-even"][j] <= 0.97
    tv[n] = {"tau_star": t, "search_gen": res["tau"][n]["search-gen"][j], "qad_even": res["tau"][n]["qad-even"][j], "pass": bool(ok)}
res["tau_verdict"] = tv; res["tau_adopt"] = bool(tv) and all(v["pass"] for v in tv.values())
p(f"\nτ verdict: {json.dumps(tv)} -> **{'ADOPT' if res['tau_adopt'] else 'not adopted'}**")

# ---------------------------------------------------------------- #125
KD = f"{D}/kl"
if os.path.isdir(KD):
    p("\n## #125 head settings (CPU proxy, 64 stratified blocks)\n")
    def e(s, v, a, K):
        tag = f"{v}_{a.replace(':', '')}_K{K}"
        f = f"{KD}/{s}/{tag}.npz"
        if not os.path.exists(f): return None
        z = np.load(f); return z["kA"].astype(np.float64) + float(z["wB"]) * z["kB"] + float(z["wC"]) * z["kC"]
    g2 = {}
    p("G2 (proxy validity), search-gen:\n")
    for K in (3, 4):
        real, c0, u = e("search", "real", "one", K), e("search", "raw", "one", K), f"u{K}"
        if real is None or c0 is None: continue
        a = mean("search-gen", real) / mean("search-gen", kl("search", u)); b = ratio("search-gen", c0, real)
        g2[K] = {"est_real_over_headsim": a, "c0_over_real": b, "pass": bool(0.85 <= a <= 1.15 and 0.80 <= b[0] <= 1.10)}
        p(f"- K{K}: KL_est(real) / headsim {u} = {a:.4f} (rule 0.85-1.15); KL_est(C0) / KL_est(real) = {fmt(b)} (rule 0.80-1.10) -> "
          f"{'PASS' if g2[K]['pass'] else 'FAIL'}")
    res["g2"] = g2; valid = len(g2) == 2 and all(v["pass"] for v in g2.values())
    sig = ("0.01", "0.1", "1")
    sel = {}
    for so in sig:
        v = [e("qad", "mr", f"two:{so}", K) for K in (3, 4)]
        if all(x is not None for x in v): sel[so] = math.sqrt(np.prod([mean("qad-odd", x) for x in v]))
    so = min(sel, key=sel.get) if sel else None
    res["sigma_out_selection"] = {"geo_qad_odd": sel, "chosen": so}
    p(f"\nσo selection (qad-odd, geo K3/K4): {json.dumps(sel)} -> σo* = {so}\n")
    arms = ([("T", "mr", f"two:{so}")] if so else []) + [("R4", "mr-r4", "one"), ("R8", "mr-r8", "one")]
    p("| arm | K | search-gen vs B | qad-even vs B | qad-odd vs B | guard search-gen | KL_est search-gen (B) |")
    p("|---|---|---|---|---|---|---|")
    res["arms"] = {}
    for K in (3, 4):
        bS, bQ = e("search", "mr", "one", K), e("qad", "mr", "one", K)
        c0S = e("search", "raw", "one", K)
        if bS is None: continue
        if c0S is not None:
            r = ratio("search-gen", bS, c0S); res["arms"][f"B/C0-K{K}"] = {"search-gen": r}
            p(f"| B (mean-row) vs C0 | {K} | {fmt(r)} | {fmt(ratio('qad-even', bQ, e('qad', 'raw', 'one', K)))} | | | {mean('search-gen', bS):.6f} |")
        for nm, v, a in arms + [(f"T{x}", "mr", f"two:{x}") for x in sig if x != so]:
            xS, xQ = e("search", v, a, K), e("qad", v, a, K)
            if xS is None or xQ is None: continue
            s_, msk = SETS["search-gen"]; gm = msk & (S["search"]["top"] >= GUARD)
            gr = float(xS[gm].sum() / bS[gm].sum())
            r = {"search-gen": ratio("search-gen", xS, bS), "qad-even": ratio("qad-even", xQ, bQ),
                 "qad-odd": ratio("qad-odd", xQ, bQ), "guard": gr}
            res["arms"][f"{nm}-K{K}"] = r
            p(f"| {nm}{'' if nm in ('T', 'R4', 'R8') else ' (report)'} {v}/{a} | {K} | {fmt(r['search-gen'])} | "
              f"{fmt(r['qad-even'])} | {fmt(r['qad-odd'])} | {gr:.3f} | {mean('search-gen', bS):.6f} |")
    verdict = {}
    for nm, _, _ in arms:
        ks = [res["arms"].get(f"{nm}-K{K}") for K in (3, 4)]
        verdict[nm] = bool(valid and all(k is not None and k["search-gen"][0] <= 0.97 and k["qad-even"][0] <= 0.97
                                         and k["guard"] <= 1.3 for k in ks))
    if verdict.get("R4") and verdict.get("R8"):
        r8 = all(res["arms"][f"R8-K{K}"]["search-gen"][0] <= 0.97 * res["arms"][f"R4-K{K}"]["search-gen"][0] for K in (3, 4))
        verdict["R_pick"] = "R8" if r8 else "R4"
    res["verdict"] = verdict; res["proxy_valid"] = valid
    p(f"\nVerdict (rule: ≤ 0.97 at K3 and K4 on search-gen and qad-even, guard ≤ 1.3; proxy valid {valid}): {json.dumps(verdict)}")
    # stratum breakdown at K3 for B, T*, R8 (search-gen)
    p("\nStratum share of KL_est (search-gen, B):")
    for K in (3, 4):
        f = f"{KD}/search/mr_one_K{K}.npz"
        if os.path.exists(f):
            z = np.load(f); s_, msk = SETS["search-gen"]; a = z["kA"][msk].mean(); b = z["wB"] * z["kB"][msk].mean(); c = z["wC"] * z["kC"][msk].mean()
            p(f"- K{K}: A {a / (a + b + c):.2f}, B {b / (a + b + c):.2f}, C {c / (a + b + c):.2f}")

open(OUT, "w").write("\n".join(L) + "\n")
json.dump(res, open(OUT[:-3] + ".json", "w"), indent=1, default=str)
