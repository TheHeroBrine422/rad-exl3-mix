#!/usr/bin/env python3
"""compare2.py DATADIR -- w63: corrected-swj arms (main, gu, noi8k7) vs w58's (copied in as w58main, w58gu, w58noi8k7).
Per size: accurate / fast predicted KLD old -> new, change, board-anchored (V2 board x KLD / V2-menu KLD), and how many
tensor picks (option, mode) differ. Fast variant = report.py's rule. Every KLD is a kfit2-model prediction."""
import collections, sys
sys.argv = [sys.argv[0], sys.argv[1]]
import importlib.util, os, io, contextlib
spec = importlib.util.spec_from_file_location("rep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.py"))
rep = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(rep)
acc, fast, V2B, pct, lv = rep.acc, rep.fast, rep.V2B, rep.pct, rep.lv

def diff(a, b):
    return sum(1 for w in a["picks"] if a["picks"][w] != b["picks"].get(w)) if "picks" in a and "picks" in b else "-"
def mk(r):
    c = collections.Counter()
    for k, v in r["param_share"].items():
        x = lv(k)
        if x: c[x[0]] += v
    return c
def anc(s, k):
    v2 = acc("v2menu", s)
    return f"{V2B[s] * k / v2['kld']:.4g}" if v2 and s in V2B else "-"
for arm in ("main", "gu"):
    print(f"\n## {arm}: w58 -> corrected swj (accurate | fast)\n")
    print("| size | acc w58 | acc new | change | anchored w58 -> new | picks changed | fast w58 | fast new | change | anchored w58 -> new | picks changed | fast vs acc (new) | mul1 share acc w58 -> new |")
    print("|---|" + "---|" * 12)
    for s in rep.SIZES:
        a0, a1, f0, f1 = acc("w58" + arm, s), acc(arm, s), fast("w58" + arm, s), fast(arm, s)
        if not (a0 and a1): continue
        row = [f"{s:g}", f"{a0['kld']:.4g}", f"{a1['kld']:.4g}", pct(a1["kld"], a0["kld"]), f"{anc(s, a0['kld'])} -> {anc(s, a1['kld'])}", str(diff(a1, a0))]
        if f0 and f1:
            row += [f"{f0['kld']:.4g}", f"{f1['kld']:.4g}", pct(f1["kld"], f0["kld"]), f"{anc(s, f0['kld'])} -> {anc(s, f1['kld'])}", str(diff(f1, f0)), pct(f1["kld"], a1["kld"])]
        else: row += ["-"] * 6
        row.append(f"{100 * mk(a0)['m1']:.1f} -> {100 * mk(a1)['m1']:.1f}%")
        print("| " + " | ".join(row) + " |")
print("\n## fast 6.5 / 7.0 without i8 K7 (noi8k7) vs main, w58 -> new\n")
for s in (6.5, 7.0):
    f0, f1, n0, n1 = fast("w58main", s), fast("main", s), fast("w58noi8k7", s), fast("noi8k7", s)
    if f0 and f1 and n0 and n1:
        print(f"- {s:g}: w58 {pct(n0['kld'], f0['kld'])}, new {pct(n1['kld'], f1['kld'])} (noi8k7 fast {n1['kld']:.4g} vs main fast {f1['kld']:.4g})")
print("\n## gate_up mean K vs the rest (accurate), w58 -> new\n")
def gk(r):   # mean ladder K over ffn_gate_up tensors (one shape, so params-weighted) and over all other trunk tensors (unweighted)
    g, o = [], []
    for w, (opt, mode) in r.get("picks", {}).items():
        x = lv(opt)
        if x and w != "output.weight": (g if w.split(".")[-2] == "ffn_gate_up" else o).append(x[1])
    return (sum(g) / len(g) if g else float("nan")), (sum(o) / len(o) if o else float("nan"))
print("| size | main w58 | main new | gu w58 | gu new |   (gate_up K / rest K, unweighted over tensors)")
print("|---|---|---|---|---|")
for s in rep.SIZES:
    c = [acc(a, s) for a in ("w58main", "main", "w58gu", "gu")]
    if all(c): print(f"| {s:g} | " + " | ".join("%.2f / %.2f" % gk(r) for r in c) + " |")
