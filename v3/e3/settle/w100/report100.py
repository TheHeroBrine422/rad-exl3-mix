#!/usr/bin/env python3
"""report100.py DATA -- w100's rules (../w100.md) on DATA/out/{colo,thor}/*.jsonl, DATA/real/*.jsonl, DATA/scan/scan.json,
DATA/w92logs/*.jsonl -> DATA/report.md and DATA/report.json. Every ratio is formed inside one job (one box)."""
import glob, json, math, os, sys
from collections import defaultdict

D = sys.argv[1]
FAM = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"]
INPUT = {"ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v"}
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
rows = [json.loads(l) for f in sorted(glob.glob(f"{D}/out/*/*.jsonl")) for l in open(f)]
sub = [r for r in rows if r["exp"] == "w100-sub1"]
hs = [r for r in rows if r["exp"] == "w100-hshape"]
md, js = [], {}
P = md.append


def cells():
    yield "all", lambda t: True, 18, 27
    yield "inputs", lambda t: t in INPUT, 10, 15
    yield "rest", lambda t: t not in INPUT, 8, 12
    for f in FAM: yield f, (lambda f: lambda t: t == f)(f), 2, 3


# ---------------------------------------------------------------- #128
P("# w100 report\n\n## #128 sub-1 hygiene\n")
P(f"sub1 rows: {len(sub)} ({len({(r['layer'], r['w'], r['sample']) for r in sub})} tensor x samples)\n")
Ks = sorted({r["bits"] for r in sub})
cur = {(r["layer"], r["w"], r["sample"], r["bits"]): r for r in sub if r["arm"] == "cur"}
arm = lambda a: {(r["layer"], r["w"], r["sample"], r["bits"]): r for r in sub if r["arm"] == a}
ext, dri = arm("ext"), arm("drift")
# 1. C9
P("### 1 · C9 statistics (cur arm; M = H_pub, after the per-output-channel refit)\n")
P("| K | family (α) | c_mean | c q50 (mean) | c_max | e_mean (clipped) | rows at clip | n | fires |")
P("|---|---|---|---|---|---|---|---|---|")
c9 = {}
for K in Ks:
    for f in ("ssm_out", "attn_output", "ffn_down"):
        rr = [r for k, r in cur.items() if k[3] == K and k[1] == f and "c9" in r]
        if not rr: continue
        cm = sum(r["c9"]["c_mean"] for r in rr) / len(rr); em = sum(r["c9"]["e_mean"] for r in rr) / len(rr)
        q50 = sum(r["c9"]["c_q01_q50_q99"][1] for r in rr) / len(rr); cx = max(r["c9"]["c_max"] for r in rr)
        clip = sum(r["c9"]["clipped"] for r in rr) / sum(r["c9"]["rows"] for r in rr)
        al = rr[0]["c9"]["alpha"]
        fire = K < 1.5 and ((al > 0 and cm > 1.3) or (al < 0 and cm < 1 / 1.3))
        c9[(K, f)] = {"c_mean": cm, "e_mean": em, "fires": fire, "n": len(rr)}
        P(f"| {K:g} | {f} ({al:g}) | {cm:.3f} | {q50:.3f} | {cx:.3f} | {em:.3f} | {clip:.1%} | {len(rr)} | {'**FIRES**' if fire else 'no'} |")
fires1 = any(v["fires"] for v in c9.values())
P(f"\n**C9 rule: {'FIRES → propose α(K) = α · clip(K − 1, 0, 1)' if fires1 else 'does not fire'}.**\n")
js["c9"] = {f"{k[0]}|{k[1]}": v for k, v in c9.items()}; js["c9_fires"] = fires1
real = [json.loads(l) for f in sorted(glob.glob(f"{D}/real/*.jsonl")) for l in open(f)]
if real:
    P("Real 27B codes (w92, i4, layers 2–3), in-job C9 recomputed here (G3 = |e_mean / job's record − 1|):\n")
    P("| K | key | α | e_mean | c_mean | c q50 | rows at clip | G3 |")
    P("|---|---|---|---|---|---|---|---|")
    for r in real:
        if "c9" in r:
            c = r["c9"]
            P(f"| {r['K']} | {r['key'].split('layers.')[1]} | {float(c.get('alpha', 0)) if 'alpha' in c else ''} | {c['e_mean']:.3f} | {c['c_mean']:.3f} | {c['c_q01_q50_q99'][1]:.3f} | {c['clipped']}/{c['rows']} | {r['G3_rel']:.1e} |")
    js["G3"] = max((r["G3_rel"] for r in real if r.get("G3_rel") is not None), default=None)
    P(f"\nG3 max: {js['G3']}\n")
# 2. g_scale edge
P("### 2 · g_scale grid edge (cur arm's search; exllamav3 grid coarse 0.1..1.9)\n")
P("| K | top edge (g 2.05) | bottom edge | g median | g min / max | fires (> 10%) | ext arm: geo rel2_B ext / cur (all, n changed) | inputs | rest |")
P("|---|---|---|---|---|---|---|---|---|")
edge = {}
for K in Ks:
    rr = [r for k, r in cur.items() if k[3] == K]
    gs = sorted(r["search"]["g_scale"] for r in rr)
    top = sum(r["search"]["edge"] == "top" for r in rr); bot = sum(r["search"]["edge"] == "bottom" for r in rr)
    fire = (top + bot) / len(rr) > 0.10
    rat = lambda sel: [ext[k]["rel2"]["B"] / r["rel2"]["B"] if k in ext else 1.0 for k, r in cur.items() if k[3] == K and sel(k[1])]
    ra, ri, rs = rat(lambda t: True), rat(lambda t: t in INPUT), rat(lambda t: t not in INPUT)
    nchg = sum(1 for k in ext if k[3] == K)
    edge[K] = {"top": top, "bottom": bot, "n": len(rr), "fires": fire, "ext_ratio": geo(ra), "ext_inputs": geo(ri), "ext_rest": geo(rs), "n_changed": nchg,
               "ext_g": sorted(r["ext_search"]["g_scale"] for k, r in cur.items() if k[3] == K)}
    P(f"| {K:g} | {top}/{len(rr)} | {bot}/{len(rr)} | {gs[len(gs)//2]:.3f} | {gs[0]:.3f} / {gs[-1]:.3f} | {'**FIRES**' if fire else 'no'} | {geo(ra):.4f} ({nchg}) | {geo(ri):.4f} | {geo(rs):.4f} |")
for K in Ks:
    g = edge[K]["ext_g"]; P(f"\nK{K:g}: ext grid's g picks: median {g[len(g)//2]:.3f}, min {g[0]:.3f}, max {g[-1]:.3f}, at the new top edge (3.9+): {sum(x >= 3.95 for x in g)}")
js["edge"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "ext_g"} for k, v in edge.items()}
w92 = [json.loads(l) for f in sorted(glob.glob(f"{D}/w92logs/*.jsonl")) for l in open(f)]
if w92:
    P("\nReal w92 job logs (i4, 13 tensors per level): " + "; ".join(
        f"K{K}: g " + ", ".join(sorted({f'{r['g_scale']:.3f}' for r in w92 if r['K'] == K})) for K in sorted({r["K"] for r in w92})))
    # amendment 1: the rule also reads the production logs (fires if > 10% at an edge)
    pe = {}
    for K in sorted({r["K"] for r in w92}):
        rr = [r for r in w92 if r["K"] == K]; top = sum(abs(r["g_scale"] - 2.05) < 1e-6 for r in rr)
        pe[str(K)] = {"top": top, "n": len(rr), "fires": top > 0.1 * len(rr)}
        P(f"\nProduction logs K{K}: {top}/{len(rr)} at the top edge (g 2.05) → {'**FIRES**' if pe[str(K)]['fires'] else 'no'}")
    js["edge_prod"] = pe
    fe = any(v["fires"] for v in pe.values()) or any(v["fires"] for v in edge.values())
    P(f"\n**g_scale rule (proxy or production > 10% at an edge): {'FIRES → propose the coarse grid to 3.9 on the top side' if fe else 'does not fire'}.**")
    js["edge_fires"] = fe
# amendment 1: fixed g at K0.125 (report only)
g125 = defaultdict(dict)
for r in rows:
    if r["exp"] == "w100-g125": g125[(r["layer"], r["w"], r["sample"])][r["arm"]] = r
if g125:
    P("\nAmendment 1 (report only): K0.125 i8 at fixed g vs the cur arm (proxy pick ~1.65); geo ratio, held-out two-sided rel2_B / one-sided rel1_pub:\n")
    P("| g | all rel2_B | all rel1_pub | inputs rel2_B | rest rel2_B | tensors×samples better (rel2_B) |")
    P("|---|---|---|---|---|---|")
    gj = {}
    for a in sorted({a for v in g125.values() for a in v}, key=lambda a: float(a[1:])):
        ks = [k for k in g125 if a in g125[k] and (k[0], k[1], k[2], 0.125) in cur]
        rb = lambda k: g125[k][a]["rel2"]["B"] / cur[(k[0], k[1], k[2], 0.125)]["rel2"]["B"]
        rp = lambda k: g125[k][a]["rel1"]["pub"] / cur[(k[0], k[1], k[2], 0.125)]["rel1"]["pub"]
        v = (geo([rb(k) for k in ks]), geo([rp(k) for k in ks]), geo([rb(k) for k in ks if k[1] in INPUT]), geo([rb(k) for k in ks if k[1] not in INPUT]),
             sum(rb(k) < 1 for k in ks))
        gj[a] = v
        P(f"| {a[1:]} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} | {v[4]}/{len(ks)} |")
    js["g125"] = gj
# 3. drift
P("\n### 3 · Measured LDLQ drift (cur arm; table value 1.08 below K1 and at K1)\n")
P("| K | D = geo d_all (all) | inputs | rest | min / max | d_fork (fork comment) | fires | drift arm: geo rel2_B drift / cur (n changed) |")
P("|---|---|---|---|---|---|---|---|")
dr = {}
for K in Ks:
    rr = [r for k, r in cur.items() if k[3] == K]
    da = [r["measured"]["d_all"] for r in rr]
    di = [r["measured"]["d_all"] for r in rr if r["w"] in INPUT]; ds = [r["measured"]["d_all"] for r in rr if r["w"] not in INPUT]
    Dk = geo(da)
    fire = (abs(Dk / 1.08 - 1) >= 0.005) if K >= 1 else (Dk >= 1.2 or Dk <= 1.0)
    rat = [dri[k]["rel2"]["B"] / r["rel2"]["B"] if k in dri else 1.0 for k, r in cur.items() if k[3] == K]
    dr[K] = {"D": Dk, "inputs": geo(di), "rest": geo(ds), "fires": fire, "drift_arm_ratio": geo(rat), "n_changed": sum(1 for k in dri if k[3] == K),
             "d_fork": geo([r["measured"]["d_fork"] for r in rr])}
    P(f"| {K:g} | {Dk:.4f} | {geo(di):.4f} | {geo(ds):.4f} | {min(da):.3f} / {max(da):.3f} | {dr[K]['d_fork']:.4f} | {'**FIRES**' if fire else 'no'} | {geo(rat):.4f} ({dr[K]['n_changed']}) |")
js["drift"] = {str(k): v for k, v in dr.items()}
# 4. zero floor
P("\n### 4 · K0.125 vs a zero tensor (rel ≥ 0.97 → price as zero / drop the cell)\n")
flags = []
if real:
    P("Real codes (w92 K0.125, i4; undamped calib H_in; Q = 0 gives 1):\n")
    P("| tensor | rel1 | rel2 (bn1920) | rel1 best row scalar | rows with e_n > 1 |")
    P("|---|---|---|---|---|")
    for r in real:
        if r["K"] != "0.125": continue
        P(f"| {r['key'].split('layers.')[1]} | {r['rel1']:.3f} | {r['rel2']:.3f} | {r['rel1_opt']:.3f} | {r['rows_rel_gt1']}/{r['rows']} |")
        if max(r["rel1"], r["rel2"]) >= 0.97: flags.append(("real", r["key"], r["rel1"], r["rel2"]))
P("\nProxy (i8 K0.125, 27 tensors, geo over 2 samples): max rel2_B / rel1_pub per family:\n")
P("| family | tensors: rel2_B (held out) / rel1_pub |")
P("|---|---|")
for f in FAM:
    ts = sorted({k[0] for k in cur if k[1] == f})
    cellv = []
    for l in ts:
        rr = [r for k, r in cur.items() if k[0] == l and k[1] == f and k[3] == 0.125]
        if not rr: continue
        a, b = geo([r["rel2"]["B"] for r in rr]), geo([r["rel1"]["pub"] for r in rr])
        cellv.append(f"L{l} {a:.3f} / {b:.3f}")
        if max(a, b) >= 0.97: flags.append(("proxy", f"{l}:{f}", a, b))
    P(f"| {f} | {'; '.join(cellv)} |")
P(f"\n**Zero-floor rule: {'FIRES for ' + str(flags) if flags else 'no tensor at rel ≥ 0.97: the K0.125 floor stands'}.**\n")
js["zero_flags"] = flags
# 5. scan
sc = json.load(open(f"{D}/scan/scan.json")) if os.path.exists(f"{D}/scan/scan.json") else []
if sc:
    P("### 5 · suh / svh scan (gpubox, read-only)\n")
    z = sum(r["zero"] for r in sc); s = sum(r["subnormal"] for r in sc); nf = sum(r["nonfinite"] for r in sc); te = sum(r["tensors"] for r in sc)
    P(f"{len(sc)} directories, {te} scale tensors: zeros {z}, subnormals {s}, non-finite {nf}.\n")
    import re
    hi = [(r["dir"], f["t"]) for r in sc for f in r["flag"] if (f["zero"] or f["subnormal"]) and
          any(float(k) >= 1 for k in (r["K"] or re.findall(r"/K([0-9.]+)(?:/|$)", r["dir"])))]
    P(f"**Scan rule (zero / subnormal at K ≥ 1): {'FIRES ' + str(hi) + ' → propose R14 clamp' if hi else 'does not fire'}.**\n")
    js["scan_fires"] = hi
    for r in sc:
        for f in r["flag"]:
            if f["zero"] or f["subnormal"]:
                P(f"- {r['dir']} K{','.join(r['K']) or '?'}: {f['t']} zero {f['zero']} subnormal {f['subnormal']} min {f['min_abs']:.2e} (median {f['median']:.2e})")
# ---------------------------------------------------------------- #129
P("\n## #129 H_out shape (σ_out × h^p), pick on (B1, O1), judge on (B2, O2)\n")
CUR = "E-so0.01-p1"
by = defaultdict(dict)
for r in hs: by[(r["layer"], r["w"], r["sample"], r["bits"])][r["arm"]] = r
arms = sorted({r["arm"] for r in hs if r["arm"].startswith("E-")})
P(f"hshape rows: {len(hs)} ({len(by)} tensor x sample x K), arms {len(arms)} + full\n")
res = {}
for K in sorted({k[3] for k in by}):
    P(f"\n### K{K:g}\n")
    P("| cell | arm* (pick X1) | g on X2 | agree | edge | proposed | swap: arm* X2 → g X1 | geo r_B,O (arm*) | full / cur (B,O) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for name, sel, need, ntot in cells():
        ks = [k for k in by if k[3] == K and sel(k[1]) and CUR in by[k]]
        if not ks: continue
        r = lambda k, a, X: by[k][a]["rel"][X] / by[k][CUR]["rel"][X]
        def best(X): return min(arms, key=lambda a: sum(math.log(r(k, a, X)) for k in ks if a in by[k]))
        a1 = best("B1|O1"); g = 1 - geo([r(k, a1, "B2|O2") for k in ks])
        tens = sorted({(k[0], k[1]) for k in ks})
        agree = sum(1 for t in tens if geo([r(k, a1, "B2|O2") for k in ks if (k[0], k[1]) == t]) < 1)
        so, px = a1.split("-so")[1].split("-p"); edgef = so in ("0.003", "0.1") or px == "0.5"
        prop = a1 != CUR and g >= 0.02 and agree >= need
        a2 = best("B2|O2"); g2 = 1 - geo([r(k, a2, "B1|O1") for k in ks])
        BO = lambda a: geo([(by[k][a]["err"]["B1|O"] + by[k][a]["err"]["B2|O"]) / (by[k][a]["ref"]["B1|O"] + by[k][a]["ref"]["B2|O"]) /
                            ((by[k][CUR]["err"]["B1|O"] + by[k][CUR]["err"]["B2|O"]) / (by[k][CUR]["ref"]["B1|O"] + by[k][CUR]["ref"]["B2|O"])) for k in ks])
        fullname = [a for a in by[ks[0]] if a.startswith("full")][0]
        res[f"{K}|{name}"] = {"arm": a1, "g": g, "agree": agree, "n": len(tens), "need": need, "edge": edgef, "proposed": prop, "swap_arm": a2, "swap_g": g2,
                              "BO": BO(a1), "full_BO": BO(fullname)}
        P(f"| {name} | {a1} | {g:+.2%} | {agree}/{len(tens)} | {'edge' if edgef else ''} | {'**YES**' if prop else 'no'} | {a2} → {g2:+.2%} | {BO(a1):.4f} | {BO(fullname):.4f} |")
    # surface on (B, O) for the all cell
    P(f"\nSurface, all 27 × 2, geo r on (B, O) = held out on both sides (rows σ_out, columns p):\n")
    P("| σ_out \\ p | " + " | ".join(sorted({a.split('-p')[1] for a in arms}, key=float)) + " |")
    P("|---|" + "---|" * len({a.split('-p')[1] for a in arms}))
    ks = [k for k in by if k[3] == K]
    for so in sorted({a.split("-so")[1].split("-p")[0] for a in arms}, key=float):
        cellsv = []
        for px in sorted({a.split("-p")[1] for a in arms}, key=float):
            a = f"E-so{so}-p{px}"
            v = geo([((by[k][a]["err"]["B1|O"] + by[k][a]["err"]["B2|O"]) / (by[k][a]["ref"]["B1|O"] + by[k][a]["ref"]["B2|O"])) /
                     ((by[k][CUR]["err"]["B1|O"] + by[k][CUR]["err"]["B2|O"]) / (by[k][CUR]["ref"]["B1|O"] + by[k][CUR]["ref"]["B2|O"])) for k in ks if a in by[k]])
            cellsv.append(f"{v:.4f}")
        P(f"| {so} | " + " | ".join(cellsv) + " |")
js["hshape"] = res
nprop = sum(v["proposed"] for v in res.values())
P(f"\n**#129: {nprop} of {len(res)} cells proposed** ({', '.join(k for k, v in res.items() if v['proposed']) or 'none'}).")
open(f"{D}/report.md", "w").write("\n".join(md) + "\n")
json.dump(js, open(f"{D}/report.json", "w"), indent=1)
print("\n".join(md))
