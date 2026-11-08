#!/usr/bin/env python3
"""report79.py W77DIR W79DIR -- ../w79.md's rule at i4 K0.125 on the grid w77 (0.0005..0.2) + w79 (0.3, 0.5, 1.0).
Same arithmetic as w77's report77.py (r = rel_X(sigma) / rel_X(sigma_cur), pick on B1, judge on B2), K0.125 only."""
import glob, json, math, os, sys

SIG = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
FAM = ["ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v", "ssm_out", "attn_output", "ffn_gate_up", "ffn_down"]
INPUT = FAM[:5]
TENS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
        "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
        "ffn_down": (2, 30, 63)}
cur = lambda w: 0.05 if w in INPUT else 0.025
R = {}
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
        for line in open(f):
            o = json.loads(line)
            if o["cb"] == "i4" and float(o["bits"]) == 0.125: R[(o["layer"], o["w"], o["sample"], float(o["sigma"]))] = o
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
r = lambda l, w, s, sg, X: R[(l, w, s, sg)]["rel"][X] / R[(l, w, s, cur(w))]["rel"][X]
P = print
miss = [(l, w, s, sg) for w in FAM for l in TENS[w] for s in (0, 1) for sg in SIG if (l, w, s, sg) not in R]
P(f"# w79 / #114 K0.125 edge report ({len(R)} K0.125 rows; missing {len(miss)})\n")
if miss: P("missing: " + ", ".join(f"{l}:{w} s{s} {sg:g}" for l, w, s, sg in miss[:20])); sys.exit(1)
P("| cell | n | current | sigma* (B1) | B2 gain vs current | agree | would pass | B2 gain vs 0.2 | B2's own pick | B curve min, within 1% |")
P("|---|---|---|---|---|---|---|---|---|---|")
curves = {}
for name, fams in [(f, [f]) for f in FAM] + [("inputs", INPUT), ("rest", FAM[5:])]:
    ts = [(l, w) for w in fams for l in TENS[w]]
    c = {X: {sg: g([r(l, w, s, sg, X) for l, w in ts for s in (0, 1)]) for sg in SIG} for X in ("B1", "B2", "B", "A")}
    curves[name] = c
    ss = min(SIG, key=lambda sg: c["B1"][sg]); gain = 1 - c["B2"][ss]
    agree = sum(g([r(l, w, s, ss, "B2") for s in (0, 1)]) < 1 for l, w in ts); n = len(ts)
    passes = ss != cur(fams[0]) and gain >= 0.02 and agree >= math.ceil(2 * n / 3)
    g02 = 1 - c["B2"][ss] / c["B2"][0.2]; m = min(c["B"].values()); bmin = min(SIG, key=lambda sg: c["B"][sg])
    w1 = [sg for sg in SIG if c["B"][sg] <= 1.01 * m]
    P(f"| {name} | {n} | {cur(fams[0]):g} | {ss:g}{' (edge)' if ss in (SIG[0], SIG[-1]) else ''} | {100 * gain:+.2f}% | {agree}/{n} | "
      f"{'yes' if passes else 'no'} | {100 * g02:+.2f}% | {min(SIG, key=lambda sg: c['B2'][sg]):g} | {bmin:g} ({m:.4f}); {w1[0]:g}–{w1[-1]:g} |")
P("\n## Curves: geo r_B (held out) per sigma, ratio to the current setting\n")
P("| cell | " + " | ".join(f"{s:g}" for s in SIG) + " | in-sample (A) min |")
P("|---|" + "---|" * len(SIG) + "---|")
for name, c in curves.items():
    am = min(SIG, key=lambda sg: c["A"][sg])
    P(f"| {name} | " + " | ".join(f"{c['B'][s]:.4f}" for s in SIG) + f" | {am:g} ({c['A'][am]:.4f}) |")
P("\n## Per tensor x sample: sigma* on B (held out)\n")
rows = [(w, l, s, min(SIG, key=lambda sg: r(l, w, s, sg, "B"))) for w in FAM for l in TENS[w] for s in (0, 1)]
P("- " + ", ".join(f"{sg:g}: {sum(x[3] == sg for x in rows)}" for sg in SIG))
P("- rising over 0.2 -> 1.0 (r_B(1.0) > r_B(0.2)): " f"{sum(r(l, w, s, 1.0, 'B') > r(l, w, s, 0.2, 'B') for w, l, s, _ in rows)} / {len(rows)}")
