#!/usr/bin/env python3
"""mkresults.py -- RESULTS.md = RESULTS.head.md with its @..@ fields computed from results.jsonl / results-v1-mul_lo.jsonl /
asm.jsonl / results-box2.jsonl, followed by report.py's tables."""
import json, collections, statistics as st, subprocess, sys, os
H = os.path.dirname(os.path.abspath(__file__))
def load(f):
    T = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l)
        if "test" in r or r["cand"] in ("empty", "read") or str(r.get("pass", "")).startswith("c"): continue
        T[(r["cand"], r["place"], r["mode"], r["K"], r["M"], r["shape"])].append(r["us"])
    return {k: st.median(v) for k, v in T.items()}
m = load(f"{H}/results.jsonl"); v1 = load(f"{H}/results-v1-mul_lo.jsonl")
def ex(c, p, mo, K, M, d=m): return d[(c, p, mo, K, M, "gate_up")] + d[(c, p, mo, K, M, "down")]
def fac(c, p, mo, K, M, d=m): return ex(c, p, mo, K, M, d) / ex("base", "-", mo, K, M, d)
def bestp(c, mo, K, M, d=m):
    ps = sorted({k[1] for k in d if k[0] == c and k[2] == mo and k[3] == K and k[4] == M})
    return min(ps, key=lambda p: ex(c, p, mo, K, M, d))
A = {}
for l in open(f"{H}/asm.jsonl"):
    r = json.loads(l)
    if (r["MM"], r["M"], r["A4"]) == (32, 1, 0): A[r["cand"]] = r
BIN = {("mad", "-"): "Mad", ("mad", "fold"): "MadF", ("mad4", "-"): "Mad4", ("mul1_i8", "-"): "Mul1I8", ("int4grid", "-"): "Int4Grid",
       ("qtip_1mad", "-"): "Qtip1Mad", ("qtip_3inst", "-"): "Qtip3Inst", ("lc1", "lds"): "Lc1Lds", ("lc1", "glb"): "Lc1Glb",
       ("cptc5", "lds"): "Cptc5Lds", ("cptc5", "glb"): "Cptc5Glb", ("cptc5", "reg"): "Cptc5Reg", ("cptc6", "lds"): "Cptc6Lds",
       ("cptc6", "glb"): "Cptc6Glb", ("cptc6", "lds+eq3"): "Cptc6Eq3", ("cptc6full", "lds"): "Cptc6FLds", ("cptc6full", "glb"): "Cptc6FGlb",
       ("cptcst6", "lds"): "St6Lds", ("cptcst6", "glb"): "St6Glb", ("cptcst8", "lds"): "St8Lds", ("cptcst8", "glb"): "St8Glb",
       ("cptcst10", "lds"): "St10Lds", ("cptcst10", "glb"): "St10Glb", ("cptcst12", "lds"): "St12Lds", ("cptcst12", "glb"): "St12Glb", ("cptcmin", "-"): "CptcMin"}
def xv(c, p):
    b = BIN.get((c, p))
    return (A[b]["valu"] - A["Base"]["valu"]) / 16 if b in A else None
TAB = {"lc1": "64 B", "cptc5": "63 B", "cptc6": "64 B", "cptc6full": "64 B stored, 127 B expanded", "cptcst10": "1 KB", "cptcst12": "4 KB"}
rows = []
for c in ["lc1", "mad", "mad4", "cptcmin", "cptc5", "cptc6full", "cptcst10", "mul1_i8", "qtip_1mad", "int4grid", "cptcst12", "qtip_3inst", "cptc6"]:
    cells = []
    for K in (1, 2, 3, 4):
        p = bestp(c, "w8a8", K, 1); cells.append(f"{fac(c, p, 'w8a8', K, 1):.2f}" + (f" [{p}]" if p != "-" else ""))
    m8 = " / ".join(f"{fac(c, bestp(c, 'w8a8', K, 8), 'w8a8', K, 8):.2f}" + (f" [{bestp(c, 'w8a8', K, 8)}]" if bestp(c, 'w8a8', K, 8) != "-" else "") for K in (1, 2))
    p0 = "lds" if (c, "lds") in BIN else "-"
    x = xv(c, p0)
    ld = " + 1 load" if c in TAB else ""
    rows.append(f"| {c} | {TAB.get(c, '–')} | {x:+.1f}{ld} | " + " | ".join(cells) + f" | {m8} |")
F = {"F1": "\n".join(rows)}
F["BASE"] = " / ".join(f"{ex('base', '-', 'w8a8', K, 1):.1f}" for K in (1, 2, 3, 4))
F["W4"] = " / ".join(f"{ex('base', '-', 'w4a4', K, 8):.1f}" for K in (1, 2, 3, 4))
F["W8"] = " / ".join(f"{ex('base', '-', 'w8a8', K, 8):.1f}" for K in (1, 2, 3, 4))
loc = []
for L, nm in ((6, "64 B"), (8, "256 B"), (10, "1 KB"), (12, "4 KB")):
    c = f"cptcst{L}"
    if (c, "lds", "w8a8", 1, 1, "down") in m:
        loc.append(f"{nm}: lds {fac(c, 'lds', 'w8a8', 1, 1):.2f}× / glb {fac(c, 'glb', 'w8a8', 1, 1):.2f}×")
F["LOC"] = "the same shift + mask + load decode with a table of " + "; ".join(loc) + " (K = 1, M = 1; the 64 B and 256 B rows are this bench's probes)."
vr = [ex(c, p, "w8a8", K, 1, v1) / ex(c, p, "w8a8", K, 1) for c, p in (("cptc5", "lds"), ("cptc6full", "lds"), ("cptcmin", "-")) for K in (1, 2, 3)]
F["V1"] = (f"{min(vr):.2f}–{max(vr):.2f}× slower (cptc5 / cptc6full / cptcmin, K 1–3, M = 1; `results-v1-mul_lo.jsonl`): cptc5 at K = 1 read "
           f"{fac('cptc5', 'lds', 'w8a8', 1, 1, {**m, **{k: v for k, v in v1.items()}}):.2f}× instead of {fac('cptc5', 'lds', 'w8a8', 1, 1):.2f}×.")
if ("base", "-", "w8a8", 1, 0, "down") in m:
    d = []
    for c, p in (("base", "-"), ("mad", "-"), ("cptc5", "lds"), ("mul1_i8", "-"), ("qtip_3inst", "-")):
        d.append(f"{c} " + " / ".join(f"{ex(c, p, 'w8a8', K, 0):.1f}" for K in (1, 2, 3, 4)))
    F["DEC"] = "µs at K = 1 / 2 / 3 / 4 (w8a8): " + "; ".join(d) + f". For scale, this bench's fused int8 GEMM at M = 256 takes {ex('base', '-', 'w8a8', 2, 256):.0f} µs an expert."
else: F["DEC"] = "(not run)"
# the rule: µs per extra VALU instruction a weight, from the table-free 24-bit candidates (mad, mad4, cptcmin)
rule = []
for K in (1, 2, 3, 4):
    v = [(ex(c, "-", "w8a8", K, 1) - ex("base", "-", "w8a8", K, 1)) / xv(c, "-") for c in ("mad", "mad4", "cptcmin")]
    rule.append(f"{st.mean(v):.1f}")
ml = (ex("mul1_i8", "-", "w8a8", 1, 1) - ex("base", "-", "w8a8", 1, 1)) / float(rule[0]) - (xv("mul1_i8", "-") - 1)
ld64 = (ex("lc1", "lds", "w8a8", 1, 1) - ex("base", "-", "w8a8", 1, 1)) - float(rule[0]) * xv("lc1", "lds")
def ldc(c, p): return ex(c, p, "w8a8", 1, 1) - ex("base", "-", "w8a8", 1, 1) - float(rule[0]) * xv(c, p)
F["RULE"] = (f"the factor columns are the data. A reading that reproduces them at M = 1 (9070 XT, an expert = 35.4 M weights): each extra "
             f"VALU instruction a weight costs {' / '.join(rule)} µs at K = 1 / 2 / 3 / 4 (from mad, mad4, cptcmin; it fades as bandwidth takes over); "
             f"a `v_mul_lo_u32` counts as ~{ml:.1f} instructions (from mul1_i8 at K = 1); a table load a weight costs about {max(0.0, ld64):.0f} µs from 64 B "
             f"(lc1: {ld64:+.1f}, inside the noise), {ldc('cptcst10', 'glb'):.0f} µs from 1 KB in global, {ldc('cptcst10', 'lds'):.0f} µs from 1 KB in LDS, "
             f"{ldc('cptcst12', 'lds'):.0f} µs from 4 KB in LDS (K = 1).")
h = open(f"{H}/RESULTS.head.md").read()
for k, v in F.items(): h = h.replace(f"@{k}@", v)
assert "@" not in h.replace("@ ", ""), [x for x in h.split() if x.startswith("@")]
tabs = subprocess.run([sys.executable, f"{H}/report.py", f"{H}/results.jsonl", "--asm", f"{H}/asm.jsonl"], capture_output=True, text=True, check=True).stdout
out = h + "## Tables (dad1, RX 9070 XT)\n\n" + tabs
b2 = f"{H}/results-box2.jsonl"
if os.path.exists(b2):
    b = load(b2)
    out += "\n## box2 card 2 (R9700, 210 W cap): expert factor over its own base, w8a8 (dad1 / box2)\n\n"
    out += "| candidate [placement] | M | " + " | ".join(f"K={K:g}" for K in (1, 1.5, 2, 2.5, 3, 4)) + " |\n|---|---|" + "---|" * 6 + "\n"
    for M in (1, 8):
        out += f"| base, µs | {M} | " + " | ".join(f"{ex('base', '-', 'w8a8', K, M):.1f} / {ex('base', '-', 'w8a8', K, M, b):.1f}" for K in (1, 1.5, 2, 2.5, 3, 4)) + " |\n"
        for c, p in [("lc1", "lds"), ("mad", "-"), ("mad", "fold"), ("mad4", "-"), ("cptcmin", "-"), ("cptc5", "lds"), ("cptc6full", "lds"), ("cptc6", "lds"), ("cptc6", "lds+eq3"),
                     ("cptc5", "reg"), ("cptcst10", "glb"), ("cptcst10", "lds"), ("cptcst12", "lds"), ("mul1_i8", "-"), ("qtip_1mad", "-"), ("int4grid", "-"), ("qtip_3inst", "-")]:
            out += f"| {c} [{p}] | {M} | " + " | ".join(f"{fac(c, p, 'w8a8', K, M):.2f} / {fac(c, p, 'w8a8', K, M, b):.2f}" for K in (1, 1.5, 2, 2.5, 3, 4)) + " |\n"
    out += "\nbox2: two timing passes, no oracle pass on the full shapes (the tiny-shape oracle passed there for every candidate and K, see `results-box2.jsonl` `shape: tiny`).\n"
open(f"{H}/RESULTS.md", "w").write(out)
print("ok", len(out))
