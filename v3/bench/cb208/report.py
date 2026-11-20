#!/usr/bin/env python3
"""report.py RESULTS.jsonl [..] [--asm asm.jsonl] [--check C.jsonl ..] -- cb208 tables (lines tagged "pass": "c*" are
exactness passes): per configuration the median over passes of each pass's
median time; expert = gate_up + down; factor = t(candidate) / t(base) of the same K, M, mode."""
import json, sys, collections, statistics as st
args = sys.argv[1:]; chk = []; asm = None
if "--asm" in args:
    i = args.index("--asm"); asm = args[i + 1]; del args[i:i + 2]
if "--check" in args:   # exactness passes: counted, their times not used
    i = args.index("--check"); chk = args[i + 1:]; args = args[:i]
files = args
T = collections.defaultdict(list); bad = 0; ntest = 0; nw = 0; ctx = collections.defaultdict(list); cfg = {}
for f in files + chk:
    for l in open(f):
        r = json.loads(l)
        ischk = f in chk or str(r.get("pass", "")).startswith("c")   # exactness passes: counted, their times not used
        if ischk and "test" not in r:
            if r["cand"] not in ("empty", "read"): ntest += r["bad"] == 0; bad += r["bad"] > 0
            continue
        if r["cand"] in ("empty", "read"):
            ctx[(r["cand"], r["shape"], r["K"], r.get("grid") if r["cand"] == "empty" else 0)].append(r.get("GBps", r["us"])); continue
        if "test" in r:
            ntest += 1; nw += r["weights"]; bad += r["bad"] != 0; continue
        if r["bad"] > 0: bad += 1
        if r["bad"] == 0: ntest += 1
        k = (r["cand"], r["place"], r["mode"], r["K"], r["M"], r["shape"])
        T[k].append(r["us"]); cfg[k] = r.get("kw", r.get("ts"))
med = {k: st.median(v) for k, v in T.items()}
spread = [max(v) / min(v) for v in T.values() if len(v) > 1]
Ks = sorted({k[3] for k in med}); Ms = sorted({k[4] for k in med}); shapes = ["gate_up", "down"]
def ex(c, p, mode, K, M):
    try: return sum(med[(c, p, mode, K, M, s)] for s in shapes)
    except KeyError: return None
cands = sorted({(k[0], k[1]) for k in med})
# best placement per candidate and mode: lowest geomean of the expert factor over K >= 1, GEMV Ms
def score(c, p, mode):
    v = [ex(c, p, mode, K, M) / ex("base", "-", mode, K, M) for K in Ks for M in Ms if K >= 1 and 0 < M < 256 and ex(c, p, mode, K, M)]
    return st.geometric_mean(v) if v else None
best = {}
for mode in ("w8a8", "w4a4"):
    for c in sorted({c for c, _ in cands}):
        sc = [(score(c, p, mode), p) for cc, p in cands if cc == c and score(c, p, mode)]
        if sc: best[(c, mode)] = sorted(sc)
order = ["raw", "base", "mad", "mad4", "mul1_i8", "int4grid", "qtip_1mad", "qtip_3inst", "lc1", "cptc5", "cptc6", "cptc6full", "cptcst6", "cptcst8", "cptcst10", "cptcst12", "cptcmin"]
print(f"timing passes per configuration: {st.median([len(v) for v in T.values()]):g}; exactness: {ntest} checks ({nw/1e9:.2f} G decoded weights compared), {bad} failed; "
      f"pass-to-pass spread of a configuration's median: median {st.median(spread):.3f}x, max {max(spread):.3f}x" if spread else f"passes 1; {ntest} checks, {bad} failed")
print("\n### Headline: routed expert (gate_up 5120->4608 + down 2304->5120), µs per expert call and (factor over base)\n")
print("| mode | candidate [placement] | K | " + " | ".join("decode pass" if m == 0 else f"M={m}" for m in Ms) + " |"); print("|---|---|---|" + "---|" * len(Ms))
for mode in ("w8a8", "w4a4"):
    for c in order:
        if (c, mode) not in best: continue
        p = best[(c, mode)][0][1]
        for K in Ks:
            if K < 1: continue
            cells = []
            for M in Ms:
                t, b = ex(c, p, mode, K, M), ex("base", "-", mode, K, M)
                cells.append("-" if t is None else f"{t:.1f}" if c == "base" else f"{t:.1f} ({t/b:.2f})")
            if any(x != "-" for x in cells): print(f"| {mode} | {c} [{p}] | {K:g} | " + " | ".join(cells) + " |")
print("\n### Placements / forms (geomean expert factor over base, K >= 1, M = 1..8; and at M = 256)\n")
print("| mode | candidate | placement | GEMV factor | GEMM factor |"); print("|---|---|---|---|---|")
for (c, mode), sc in sorted(best.items(), key=lambda x: (x[0][1] != "w8a8", order.index(x[0][0]))):
    for s, p in sc:
        g = [ex(c, p, mode, K, 256) / ex("base", "-", mode, K, 256) for K in Ks if K >= 1 and ex(c, p, mode, K, 256)]
        print(f"| {mode} | {c} | {p} | {s:.3f} | {st.geometric_mean(g):.3f} |" if g else f"| {mode} | {c} | {p} | {s:.3f} | - |")
print("\n### Per shape, M = 1 (µs, factor over base)\n")
print("| mode | candidate | shape | " + " | ".join(f"K={K:g}" for K in Ks) + " |"); print("|---|---|---|" + "---|" * len(Ks))
for mode in ("w8a8", "w4a4"):
    for c in order:
        if (c, mode) not in best: continue
        p = best[(c, mode)][0][1]
        for s in shapes:
            cells = []
            for K in Ks:
                t, b = med.get((c, p, mode, K, 1, s)), med.get(("base", "-", mode, K, 1, s))
                cells.append("-" if t is None else f"{t:.1f}" if c == "base" else f"{t:.1f} ({t/b:.2f})")
            print(f"| {mode} | {c} [{p}] | {s} | " + " | ".join(cells) + " |")
print("\n### Context\n")
for s in shapes:
    for K in Ks:
        rd = ctx.get(("read", s, K, 0)); em = ctx.get(("empty", s, K, 144))
        if rd: print(f"- {s} K={K:g}: pure read {st.median(rd):.0f} GB/s; empty launch {st.median(em):.2f} µs")

if asm:
    A = collections.defaultdict(dict)
    for l in open(asm):
        r = json.loads(l); A[(r["MM"], r["M"], r["A4"])][r["cand"]] = r
    keys = [k for k in [(32, 1, 0), (40, 1, 0), (32, 8, 0), (32, 1, 1)] if k in A]
    print("\n### Instructions of the GEMV inner loop (device asm; one step = 16 weights a lane)\n")
    print("VALU = vector ALU instructions per weight (in brackets: over base); loads = table loads per weight "
          "(LDS or global); the loop's weight / activation loads are the same for all but [fold] (activations 4x / 8x).\n")
    print("| binary | " + " | ".join(f"K={k[0]/16:g} M={k[1]} {'w4a4' if k[2] else 'w8a8'}: VALU" for k in keys) + " | table loads |"); print("|---|" + "---|" * (len(keys) + 1))
    for c in A[keys[0]]:
        cells = []
        for k in keys:
            r, b = A[k].get(c), A[k]["Base"]
            cells.append("-" if r is None else f"{r['valu']/16:.2f} ({(r['valu']-b['valu'])/16:+.2f})")
        r, b = A[keys[0]][c], A[keys[0]]["Base"]
        tl = r["load_lds"] / 16 + max(0, r["load_global"] - b["load_global"]) / 16 if "F" != c[-1:] or c.startswith("Cptc") else 0
        print(f"| {c} | " + " | ".join(cells) + f" | {tl:.0f} |")
