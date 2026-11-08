#!/usr/bin/env python3
"""report87.py W87_DIR [--w77 W77_OUT_DIR] -- applies ../w87.md's rule (#114-03) to settle87.py outputs (*.jsonl under
W87_DIR, recursively) and prints markdown. A cell is read only when complete (every tensor x sample x sigma at its
level). --w77: w77's no-H_out rows (data/w77/out) for the side-by-side curves (same tensors, samples, sigma points;
ratios to the same current sigma)."""
import glob, json, math, os, sys

args = sys.argv[1:]; w77 = None
if "--w77" in args: i = args.index("--w77"); w77 = args[i + 1]; del args[i:i + 2]
SIG = [0.001, 0.005, 0.025, 0.05, 0.1, 0.2]
FAM = ["ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v", "ssm_out", "attn_output", "ffn_gate_up", "ffn_down"]
INPUT = FAM[:5]
TENS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
        "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
        "ffn_down": (2, 30, 63)}
LEVELS = [("i4", 0.125), ("i4", 0.5), ("i4", 1.0), ("i8", 2.0), ("i8", 2.5), ("i8", 4.0)]


def cur(w, K):
    if K <= 0.125: return 0.2
    return (0.05 if K < 1 else 0.005) if w in INPUT else 0.025


def load(d, key="rel"):
    R = {}
    for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
        for line in open(f):
            o = json.loads(line)
            R[(o["layer"], o["w"], o["sample"], o["cb"], float(o["bits"]), float(o["sigma"]))] = o
    return R


R = load(args[0]); R77 = load(w77) if w77 else {}
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def r(t, s, lv, sg, X, src=None, field="rel"):
    src = R if src is None else src
    (l, w), (cb, K) = t, lv
    a, b = src.get((l, w, s, cb, K, sg)), src.get((l, w, s, cb, K, cur(w, K)))
    return None if a is None or b is None else a[field][X] / b[field][X]


def need_of(fams, lv): return [((l, w), s, lv) for w in fams for l in TENS[w] for s in (0, 1)]


def curve(need, X, src=None, field="rel"):
    out = {}
    for sg in SIG:
        v = [r(t, s, lv, sg, X, src, field) for t, s, lv in need]
        out[sg] = None if None in v else g(v)
    return out


def judge(fams, lv):
    ts = [(l, w) for w in fams for l in TENS[w]]; need = need_of(fams, lv)
    if any(r(t, s, lv_, sg, "B") is None for t, s, lv_ in need for sg in SIG): return None
    c1 = curve(need, "B1"); sstar = min(SIG, key=lambda sg: c1[sg])
    c2 = curve(need, "B2"); gain = 1 - c2[sstar]
    agree = sum(g([r(t, s, lv, sstar, "B2") for s in (0, 1)]) < 1 for t in ts)
    curr = {cur(w, lv[1]) for _, w in ts}
    thr = math.ceil(2 * len(ts) / 3)
    prop = sstar not in curr and gain >= 0.02 and agree >= thr
    edge = sstar == SIG[0] or (sstar == SIG[-1] and SIG[-1] not in curr)
    return {"sstar": sstar, "gain": gain, "agree": agree, "n": len(ts), "thr": thr, "propose": prop, "cur": sorted(curr),
            "B": curve(need, "B"), "A": curve(need, "A"), "B1pick": sstar, "B2pick": min(SIG, key=lambda sg: c2[sg]),
            "one": curve(need, "B", field="rel1"), "w77": curve(need, "B", R77) if R77 else None, "edge": edge}


def shape(c):
    if c is None or None in c.values(): return "n/a"
    m = min(c.values()); best = min(c, key=c.get); within = [sg for sg in SIG if c[sg] <= m * 1.01]
    return f"min {best:g} ({m:.4f}); ≤1%: {within[0]:g}–{within[-1]:g}"


fmt = lambda c: " | ".join("n/a" if c is None or c[s] is None else f"{c[s]:.4f}" for s in SIG)
lab = lambda lv: f"{lv[0]} K{lv[1]:g}"
out = []; P = out.append
P(f"# w87 / #114-03 report ({len(R)} rows; w77 rows for comparison: {len(R77)})\n")
P("## Rule cells (pick on B1, judge on B2; propose: sigma* != current, B2 gain >= 2%, >= 2/3 of tensors agree)\n")
P("| level | cell | n | current | sigma* (B1) | B2 gain | agree | proposed | B2's own pick | B curve (two-sided) |")
P("|---|---|---|---|---|---|---|---|---|---|")
res = {}; nprop = 0; ncell = 0
for lv in LEVELS:
    for name, fams in [(f, [f]) for f in FAM] + [("inputs", INPUT), ("rest", FAM[5:])]:
        j = judge(fams, lv); res[(lv, name)] = j
        if j is None: P(f"| {lab(lv)} | {name} | | | incomplete | | | | | |"); continue
        ncell += 1; nprop += j["propose"]
        P(f"| {lab(lv)} | {name} | {j['n']} | {','.join(f'{x:g}' for x in j['cur'])} | {j['sstar']:g}{' (edge)' if j['edge'] else ''} | "
          f"{100 * j['gain']:+.2f}% | {j['agree']}/{j['n']} | {'**PROPOSE**' if j['propose'] else 'no'} | {j['B2pick']:g} | {shape(j['B'])} |")
P(f"\nCells read: {ncell} / {len(LEVELS) * 11}; proposed: {nprop}. B1 / B2 picks agree in "
  f"{sum(1 for j in res.values() if j and j['B1pick'] == j['B2pick'])} of {ncell}.\n")
P("## Curves: geo r_B per sigma (ratio to the current sigma), two-sided with H_out; in-sample A; one-sided score of the same quants; w77 (no H_out)\n")
P("| level | cell | curve | " + " | ".join(f"{s:g}" for s in SIG) + " | shape |")
P("|---|---|---|" + "---|" * len(SIG) + "---|")
for (lv, name), j in res.items():
    if j is None: continue
    P(f"| {lab(lv)} | {name} | **B, H_out** | {fmt(j['B'])} | {shape(j['B'])} |")
    if name in ("inputs", "rest"):
        P(f"| | | A (in-sample) | {fmt(j['A'])} | {shape(j['A'])} |")
        P(f"| | | B one-sided | {fmt(j['one'])} | {shape(j['one'])} |")
        if j["w77"] is not None: P(f"| | | B, w77 no H_out | {fmt(j['w77'])} | {shape(j['w77'])} |")
# absolute levels at sigma_cur: two-sided rel_B, one-sided rel_B of the same quant, w77's one-sided quant rel_B
P("\n## Absolute held-out error at the current sigma (geo over the group's tensors x samples)\n")
P("| level | group | rel_B two-sided (H_out quant) | rel_B one-sided (H_out quant) | rel_B one-sided (w77, no H_out quant) |")
P("|---|---|---|---|---|")
for lv in LEVELS:
    for name, fams in (("inputs", INPUT), ("rest", FAM[5:])):
        need = need_of(fams, lv)
        a = [R.get((l, w, s, lv[0], lv[1], cur(w, lv[1]))) for (l, w), s, _ in need]
        b = [R77.get((l, w, s, lv[0], lv[1], cur(w, lv[1]))) for (l, w), s, _ in need]
        if None in a: continue
        P(f"| {lab(lv)} | {name} | {g([x['rel']['B'] for x in a]):.5f} | {g([x['rel1']['B'] for x in a]):.5f} | "
          f"{'n/a' if None in b else format(g([x['rel']['B'] for x in b]), '.5f')} |")
# sigma* spread per tensor x sample (held out B)
P("\n## Per tensor x sample: held-out (B) sigma* counts per level\n")
for lv in LEVELS:
    cnt = {s: 0 for s in SIG}; n = 0
    for w in FAM:
        for l in TENS[w]:
            for sm in (0, 1):
                c = [r((l, w), sm, lv, sg, "B") for sg in SIG]
                if None in c: continue
                n += 1; cnt[SIG[min(range(len(SIG)), key=lambda i: c[i])]] += 1
    if n: P(f"- {lab(lv)} ({n}): " + ", ".join(f"{s:g} ×{cnt[s]}" for s in SIG))
print("\n".join(out))
