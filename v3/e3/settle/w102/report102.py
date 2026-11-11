#!/usr/bin/env python3
"""report102.py DATA [--allow-partial] -- w102 #136-01 / -02 (../w102.md) from DATA/{d3,b8}/*.jsonl (copies of the boxes'
out/ dirs). Gate G0 first (arm F of d3 / arm exact of b8 against w95 / w87's stored i8 rows, ../data/w95, ../data/w87,
1e-6 relative); no ratio is printed unless G0 passes on every row present. Then the rules: R_ntry = geo T16 / T1,
R_chain = geo O4 / O1 (d3, per K); R = geo L8 / fp16 (b8, per K); tensor-bootstrap 95% CI (10,000 draws, seed 102),
per family / group, the reported-only ratios, and the trunk bpw saved by 8-bit scales (shapes in SHAPES)."""
import glob, json, math, os, sys
from collections import defaultdict

import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); SD = os.path.join(H, "..", "data")
DATA = sys.argv[1]; PARTIAL = "--allow-partial" in sys.argv
geo = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
# Qwen3.8-27B trunk tensor shapes (N out, Kd in) and count (64 layers: 48 GDN + 16 attention); ffn on every layer
SHAPES = {"ssm_qkv": ((10240, 5120), 48), "ssm_z": ((6144, 5120), 48), "ssm_out": ((5120, 6144), 48),
          "attn_qg": ((12288, 5120), 16), "attn_k": ((1024, 5120), 16), "attn_v": ((1024, 5120), 16),
          "attn_output": ((5120, 6144), 16), "ffn_gate": ((17408, 5120), 64), "ffn_up": ((17408, 5120), 64),
          "ffn_down": ((5120, 17408), 64)}


def stored(exp, layer, w, K, sig):
    """w95's (K < 2) or w87's (K >= 2) stored i8 row for sample 0"""
    pat = (f"{SD}/w95/out/*/k{K:g}/{layer}-{w}-s0-d{sig:g}.jsonl" if K < 2 else f"{SD}/w87/out/*/{layer}-{w}-s0-d{sig:g}.jsonl")
    for f in glob.glob(pat):
        for ln in open(f):
            o = json.loads(ln)
            if o["cb"] == "i8" and abs(o["bits"] - K) < 1e-9: return o["rel"]["B"]
    return None


def load(item):
    rows = []
    for f in sorted(glob.glob(f"{DATA}/{item}/*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


def boot(r, n=10000):
    rng = np.random.default_rng(102); v = np.log(np.array(r)); i = rng.integers(0, len(v), (n, len(v)))
    m = np.exp(v[i].mean(1)); return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def g0(rows, arm_rel):
    bad, n, worst = [], 0, 0.0
    for o in rows:
        ref = stored(o["exp"], o["layer"], o["w"], o["bits"], o["sigma"]); mine = arm_rel(o)
        if ref is None: bad.append((o["layer"], o["w"], o["bits"], "no stored row")); continue
        d = abs(mine / ref - 1); worst = max(worst, d); n += 1
        if d > 1e-6: bad.append((o["layer"], o["w"], o["bits"], d))
    return n, worst, bad


def table(by, name):
    print(f"\n{name}: K | n | geo | 95% CI (tensor bootstrap) | inputs | rest")
    for K in sorted(by):
        r = by[K]; v = [x[0] for x in r]
        lo, hi = boot(v)
        gi = [x[0] for x in r if x[1] == "input"]; gr = [x[0] for x in r if x[1] == "rest"]
        print(f"  K{K:g} | {len(v)} | {geo(v):.4f} | [{lo:.4f}, {hi:.4f}] | {geo(gi) if gi else float('nan'):.4f} | {geo(gr) if gr else float('nan'):.4f}")


def fam(by, name):
    print(f"\n{name} per family (geo over layers):")
    for K in sorted(by):
        f = defaultdict(list)
        for x in by[K]: f[x[2]].append(x[0])
        print(f"  K{K:g}: " + ", ".join(f"{k} {geo(v):.4f}" for k, v in sorted(f.items())))


def d3():
    rows = load("d3")
    if not rows: print("d3: no rows"); return
    by = defaultdict(dict)
    for o in rows: by[(o["layer"], o["w"], o["bits"])][o["arm"]] = o
    full = {k: v for k, v in by.items() if len(v) == 6}
    print(f"#136-01 d3: {len(full)} complete (tensor, K) of {len(by)}; tensors per K: "
          + ", ".join(f"K{K:g} {sum(1 for k in full if k[2] == K)}" for K in sorted({k[2] for k in full})))
    n, worst, bad = g0([v["F"] for v in full.values()], lambda o: o["rel"]["B"])
    print(f"G0 (F vs w95 i8 rows): {n} rows, max rel diff {worst:.2e}, {len(bad)} bad -> {'PASS' if not bad and n else 'FAIL'}")
    if bad: print("  ", bad[:10]); return
    R = {"T16/T1": ("T16", "T1"), "O4/O1": ("O4", "O1"), "T1/F": ("T1", "F"), "O1/O0": ("O1", "O0"), "O4/T16(cross)": ("O4", "T16"),
         "T16/F": ("T16", "F")}
    for nm, (a, b) in R.items():
        bk = defaultdict(list)
        for (l, w, K), v in full.items(): bk[K].append((v[a]["rel"]["B"] / v[b]["rel"]["B"], v[a]["group"], w))
        table(bk, nm)
        if nm in ("T16/T1", "O4/O1", "T1/F"): fam(bk, nm)
    print("\nRule: NTRY 16 adopted at K if geo T16/T1 <= 0.995; chained if geo O4/O1 <= 0.95")
    for K in sorted({k[2] for k in full}):
        t = geo([v["T16"]["rel"]["B"] / v["T1"]["rel"]["B"] for k, v in full.items() if k[2] == K])
        c = geo([v["O4"]["rel"]["B"] / v["O1"]["rel"]["B"] for k, v in full.items() if k[2] == K])
        print(f"  K{K:g}: R_ntry {t:.4f} -> {'PASS (propose NTRY 16)' if t <= 0.995 else 'fail (NTRY 1 stays)'}; "
              f"R_chain {c:.4f} -> {'PASS' if c <= 0.95 else 'fail (not adopted)'}")
    for K in sorted({k[2] for k in full}):
        s = defaultdict(list)
        for k, v in full.items():
            if k[2] == K:
                for a in v: s[a].append(v[a]["sec"])
        print(f"  CPU sec per arm K{K:g} (median): " + ", ".join(f"{a} {np.median(x):.0f}" for a, x in s.items()))
    for K in sorted({k[2] for k in full}):     # B1 / B2 / A agreement for the NTRY ratio
        print(f"  K{K:g} T16/T1 on B1 / B2 / A: " + " / ".join(
            f"{geo([v['T16']['rel'][h] / v['T1']['rel'][h] for k, v in full.items() if k[2] == K]):.4f}" for h in ("B1", "B2", "A")))


def b8():
    rows = load("b8")
    if not rows: print("b8: no rows"); return
    print(f"\n#136-02 b8: {len(rows)} (tensor, K) rows")
    n, worst, bad = g0(rows, lambda o: o["arms"]["exact"]["rel"]["B"])
    n2, worst2, bad2 = g0(rows, lambda o: o["rel_base"]["B"])
    print(f"G0 (exact arm vs w87 / w95 i8 rows): {n} rows, max rel diff {worst:.2e} (rel_base {worst2:.2e}), {len(bad)} bad -> "
          f"{'PASS' if not bad and n else 'FAIL'}")
    if bad: print("  ", bad[:10]); return
    for arm in ("L8", "fp16", "L8su", "L8sv", "L9", "E4M3", "L8r"):
        ref = "exact" if arm == "fp16" else "fp16"
        bk = defaultdict(list)
        for o in rows: bk[o["bits"]].append((o["arms"][arm]["rel"]["B"] / o["arms"][ref]["rel"]["B"], o["group"], o["w"]))
        table(bk, f"{arm}/{ref}")
        if arm == "L8": fam(bk, "L8/fp16")
    print("\nRule: pass at K if geo L8/fp16 <= 1.001")
    for K in sorted({o["bits"] for o in rows}):
        r = geo([o["arms"]["L8"]["rel"]["B"] / o["arms"]["fp16"]["rel"]["B"] for o in rows if o["bits"] == K])
        print(f"  K{K:g}: R {r:.5f} -> {'PASS' if r <= 1.001 else 'fail'}")
    print("  log2 step (su / sv), median over tensors: " + ", ".join(
        f"{np.median([math.log2(o['su_range'][1] / o['su_range'][0]) / 127 for o in rows]):.4f} / "
        f"{np.median([math.log2(o['sv_range'][1] / o['sv_range'][0]) / 127 for o in rows]):.4f}" for _ in [0]))
    tot_w = sum(n * k * c for (n, k), c in SHAPES.values()); sv = sum(8 * (n + k) * c for (n, k), c in SHAPES.values())
    print(f"  trunk bpw saved by 8-bit su / sv (fp16 -> 8 bit, every trunk tensor): {sv / tot_w:.5f} bpw "
          f"(trunk {tot_w / 1e9:.2f}B weights); per family: " + ", ".join(f"{f} {8 * (n + k) / (n * k):.5f}" for f, ((n, k), c) in SHAPES.items()))


if __name__ == "__main__":
    d3(); b8()
