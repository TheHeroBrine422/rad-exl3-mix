#!/usr/bin/env python3
"""report77.py DIR... [--gate gate77.json] -- applies ../w77.md's rules to settle77.py outputs (any *.jsonl under DIR,
recursively; both boxes' out/ copied into data/) and prints markdown. A cell is read only when complete (every tensor
x sample x level x sigma). Lead 11-07: grid + 0.0005 / 0.001 (0.0005 winning = an edge); i4 K0.125 reported separately
as a bound (own sigma* per family, flags: sigma* at 0.2, held-out error rising with sigma everywhere, scale collapse)."""
import collections, glob, json, math, os, sys

args = sys.argv[1:]; gate = None
if "--gate" in args: i = args.index("--gate"); gate = json.load(open(args[i + 1])); del args[i:i + 2]
SIG = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
FAM = ["ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v", "ssm_out", "attn_output", "ffn_gate_up", "ffn_down"]
INPUT = FAM[:5]
TENS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
        "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
        "ffn_down": (2, 30, 63)}
BANDS = {"sub-1": [("i4", 0.25), ("i4", 0.5), ("i4", 0.75)], "K1": [("i4", 1.0)], "K1.5": [("i8", 1.5)],
         "K2-i8": [("i8", 2.0)], "K2-i4": [("i4", 2.0)], "K2.5": [("i8", 2.5)]}
BOUND = {"K0.125 (bound)": [("i4", 0.125)]}
cur = lambda w, K: (0.05 if K < 1 else 0.005) if w in INPUT else 0.025
R = {}
for d in args:
    for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
        for line in open(f):
            o = json.loads(line)
            R[(o["layer"], o["w"], o["sample"], o["cb"], float(o["bits"]), float(o["sigma"]))] = o
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def r(t, s, lv, sg, X):
    (l, w), (cb, K) = t, lv
    a, b = R.get((l, w, s, cb, K, sg)), R.get((l, w, s, cb, K, cur(w, K)))
    return None if a is None or b is None else a["rel"][X] / b["rel"][X]


def cell(fams, band):
    ts = [(l, w) for w in fams for l in TENS[w]]
    need = [(t, s, lv) for t in ts for s in (0, 1) for lv in {**BANDS, **BOUND}[band]]
    ok = all(r(t, s, lv, sg, "B") is not None for t, s, lv in need for sg in SIG)
    return ts, need, ok


def curve(need, X): return {sg: g([r(t, s, lv, sg, X) for t, s, lv in need]) for sg in SIG}


def judge(fams, band, thr_agree):
    ts, need, ok = cell(fams, band)
    if not ok: return None
    c1 = curve(need, "B1"); sstar = min(SIG, key=lambda sg: c1[sg])
    c2 = curve(need, "B2"); gain = 1 - c2[sstar]
    LV = {**BANDS, **BOUND}[band]
    agree = sum(g([r(t, s, lv, sstar, "B2") for s in (0, 1) for lv in LV]) < 1 for t in ts)
    curr = {cur(w, K) for _, w in ts for _, K in LV}
    adopt = sstar not in curr and gain >= 0.02 and agree >= thr_agree(len(ts))
    pick2 = min(SIG, key=lambda sg: c2[sg])
    return {"sstar": sstar, "gain": gain, "agree": agree, "n": len(ts), "adopt": adopt, "cur": sorted(curr),
            "B": curve(need, "B"), "A": curve(need, "A"), "pub": curve(need, "pub"), "pick_B2": pick2,
            "edge": sstar in (SIG[0], SIG[-1])}


def shape(c):
    m = min(c.values()); best = min(c, key=c.get)
    within = [sg for sg in SIG if c[sg] <= m * 1.01]
    return f"min {best:g} ({m:.4f}); within 1%: {within[0]:g}–{within[-1]:g}"


out = []
P = out.append
P(f"# w77 / #114 report ({len(R)} rows)\n")
if gate: P(f"Gate (i4 sub-1): **{gate['verdict']}** (decode {gate['PASS_a']}, search {gate['PASS_b']}).\n")
P("## Rule cells (pick on B1, judge on B2; adopt: sigma* != current, gain >= 2%, >= 2/3 tensors agree)\n")
P("| band | cell | n | current | sigma* (B1) | B2 gain | agree | adopt | B2's own pick | B curve shape |")
P("|---|---|---|---|---|---|---|---|---|---|")
res = {}
for band in list(BANDS) + list(BOUND):
    if band in BOUND: P(f"| **{band}** (reported, not part of the ladder rule) | | | | | | | | | |")
    for name, fams, thr in [(f, [f], lambda n: math.ceil(2 * n / 3)) for f in FAM] + \
                           [("inputs", INPUT, lambda n: math.ceil(2 * n / 3)), ("rest", FAM[5:], lambda n: math.ceil(2 * n / 3))]:
        j = judge(fams, band, thr); res[(band, name)] = j
        if j is None: P(f"| {band} | {name} | | | incomplete | | | | | |"); continue
        P(f"| {band} | {name} | {j['n']} | {','.join(f'{x:g}' for x in j['cur'])} | {j['sstar']:g}{' (edge)' if j['edge'] else ''} | "
          f"{100 * j['gain']:+.2f}% | {j['agree']}/{j['n']} | {('would pass (bound: report only)' if band in BOUND else '**ADOPT**') if j['adopt'] else 'no'} | {j['pick_B2']:g} | {shape(j['B'])} |")
P("\n## Curves: geo r_B (held out, B1 + B2) per sigma, ratio to the current setting\n")
P("| band | cell | " + " | ".join(f"{s:g}" for s in SIG) + " | in-sample (A) min | pub min |")
P("|---|---|" + "---|" * len(SIG) + "---|---|")
for (band, name), j in res.items():
    if j is None: continue
    amin = min(j["A"], key=j["A"].get); pmin = min(j["pub"], key=j["pub"].get)
    P(f"| {band} | {name} | " + " | ".join(f"{j['B'][s]:.4f}" for s in SIG) + f" | {amin:g} ({j['A'][amin]:.4f}) | {pmin:g} ({j['pub'][pmin]:.4f}) |")
P("\n## Sub-1 per level (geo r_B per sigma, groups)\n")
P("| level | group | " + " | ".join(f"{s:g}" for s in SIG) + " |")
P("|---|---|" + "---|" * len(SIG))
for lv in BANDS["sub-1"]:
    for name, fams in (("inputs", INPUT), ("rest", FAM[5:])):
        need = [((l, w), s, lv) for w in fams for l in TENS[w] for s in (0, 1)]
        if all(r(t, s, lv_, sg, "B") is not None for t, s, lv_ in need for sg in SIG):
            c = curve(need, "B"); P(f"| {lv[0]} K{lv[1]:g} | {name} | " + " | ".join(f"{c[s]:.4f}" for s in SIG) + " |")
# K0.125 bound: flags
P("\n## K0.125 bound: flags (per tensor x sample)\n")
lv = BOUND["K0.125 (bound)"][0]; rows = []
for w in FAM:
    for l in TENS[w]:
        for sm in (0, 1):
            c = [r((l, w), sm, lv, sg, "B") for sg in SIG]
            a = R.get((l, w, sm, "i4", 0.125, cur(w, 0.125))); b = R.get((l, w, sm, "i4", 0.25, cur(w, 0.25)))
            if None in c or a is None or b is None: continue
            rising = all(x <= y for x, y in zip(c, c[1:])); best = SIG[min(range(len(SIG)), key=lambda i: c[i])]
            rows.append((w, l, sm, best, rising, a["s_exl"] / b["s_exl"], a["rel"]["B"], b["rel"]["B"]))
if rows:
    P(f"- tensors x samples read: {len(rows)}; sigma* at 0.2: {sum(x[3] == 0.2 for x in rows)}; at 0.0005: "
      f"{sum(x[3] == 0.0005 for x in rows)}; held-out error rising with sigma over the whole grid: {sum(x[4] for x in rows)}")
    P("- scale (s_exl K0.125 / K0.25) range: " f"{min(x[5] for x in rows):.3f}..{max(x[5] for x in rows):.3f}; "
      f"rel_B at sigma_cur K0.125 range {min(x[6] for x in rows):.3f}..{max(x[6] for x in rows):.3f} "
      f"(K0.25: {min(x[7] for x in rows):.3f}..{max(x[7] for x in rows):.3f})")
    odd = [x for x in rows if x[3] == 0.2 or x[5] < 0.5 or x[5] > 2 or x[6] >= 0.95]
    P("- odd (sigma* = 0.2, scale ratio outside [0.5, 2], or rel_B >= 0.95): " +
      (", ".join(f"{l}:{w} s{sm} (sigma* {b:g}, scale x{sr:.2f}, rel_B {rb:.3f})" for w, l, sm, b, _, sr, rb, _ in odd) or "none"))
# item 2
P("\n## Item 2: #38's K1.5 / K2.5 verdict (w43's rule: gain >= 1% at both K, every tensor r_B <= 1.00 on both samples)\n")
P("| group | sigma | K1.5 gain (B) | K2.5 gain (B) | worst tensor r_B | K1.5 / K2.5 gain in-sample (A) | passes |")
P("|---|---|---|---|---|---|---|")
for name, fams in (("inputs", INPUT), ("rest", FAM[5:])):
    best = None; inc = False
    for sg in SIG:
        row, worst, gA, okall = [], 0, [], True
        for K in (1.5, 2.5):
            need = [((l, w), s, ("i8", K)) for w in fams for l in TENS[w] for s in (0, 1)]
            vals = [r(t, s, lv, sg, "B") for t, s, lv in need]
            if None in vals: okall = None; break
            row.append(1 - g(vals)); worst = max(worst, max(vals)); gA.append(1 - g([r(t, s, lv, sg, "A") for t, s, lv in need]))
        if okall is None: P(f"| {name} | {sg:g} | incomplete | | | | |"); inc = True; continue
        if sg == (0.005 if name == "inputs" else 0.025): continue
        ps = all(x >= 0.01 for x in row) and worst <= 1.0
        if ps and (best is None or sum(row) > best[1]): best = (sg, sum(row))
        P(f"| {name} | {sg:g} | {100 * row[0]:+.2f}% | {100 * row[1]:+.2f}% | {worst:.4f} | {100 * gA[0]:+.2f}% / {100 * gA[1]:+.2f}% | {'PASS' if ps else 'no'} |")
    P(f"| {name} | verdict | | | | | {'incomplete' if inc else 'replace with ' + format(best[0], 'g') if best else 'keep (holds)'} |")
print("\n".join(out))
