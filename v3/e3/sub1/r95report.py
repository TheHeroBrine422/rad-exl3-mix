#!/usr/bin/env python3
"""r95report.py [r95.jsonl] -- #95 numbers and verdicts (README "#95", rules fixed before data).

95-01 act order: ratio = rel(hdiag) / rel(natural), LDLQ stage, per tensor; median / p90 over the 10 tensors per K
      (and per family). Rule: pass if median <= 0.99 at >= 2 of K2, 2.5, 3, 4, 5 and > 1.005 at none.
95-02 bf16 (row, 128) scales (+0.125 bpw): refit rel at K vs the plain trellis at equal total bits K + 0.125, read
      log-linearly between K and the next quarter step K + 0.25: rel_eq = sqrt(rel(K) rel(K + 0.25)). Also shown: vs
      K itself (gross gain) and vs K + 0.25 (twice its bits). Rule (on rs128, the per-block closed form as specified):
      worth a 0.8B KL test if the median of rel_rs / rel_eq is <= 0.97 on attn_output and on ssm_out at K5 or K6.
      rs128j (joint per-row solve under the full H) and ffn_down are reported only."""
import json, math, sys
from collections import defaultdict
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/data/work/e3t/r95.jsonl"
R = {}
for l in open(path):
    try: o = json.loads(l)
    except ValueError: continue
    R[(o["layer"], o["w"], o.get("order"), o["method"], float(o["bits"]))] = o   # last line wins (reruns)
CB = "cubic0.02_i8"
T1 = [(10, "ssm_qkv"), (10, "ssm_out"), (10, "ffn_gate_up"), (10, "ffn_down"), (30, "ffn_down"), (50, "ssm_qkv"),
      (50, "ffn_gate_up"), (19, "attn_qg"), (19, "attn_output"), (43, "attn_v")]
T2 = {"attn_output": [3, 19, 43, 63], "ssm_out": [2, 10, 30, 50, 62], "ffn_down": [10, 30]}


def rel(l, w, K, method="ldlq", order="natural"):
    o = R.get((l, w, order, f"{method}_{CB}", float(K)))
    return o["rel"] if o else None


def stats(v):
    return f"median {np.median(v):.4f}  p90 {np.percentile(v, 90):.4f}  min {min(v):.4f}  max {max(v):.4f}  n={len(v)}"


print(f"# #95 report from {path} ({len(R)} lines)\n")
print("## 95-01 act order (hdiag / natural, LDLQ stage)\n")
K1 = [2, 2.5, 3, 4, 5]; med = {}; complete = True
for K in K1:
    r = {}; cv = []
    for l, w in T1:
        a, b = rel(l, w, K, order="hdiag"), rel(l, w, K)
        if a is not None and b is not None: r[(l, w)] = a / b
        o = R.get((l, w, "hdiag", f"ldlq_{CB}", float(K)))
        if o: cv.append(o["mass_cv"])
    if len(r) < len(T1): complete = False
    if not r: print(f"K{K:g}: no data"); continue
    med[K] = float(np.median(list(r.values())))
    print(f"K{K:g}: {stats(list(r.values()))}   (sub-block H-mass CV, mean {np.mean(cv):.4f})")
    fam = defaultdict(list)
    for (l, w), v in r.items(): fam[w].append(v)
    print("   " + "  ".join(f"{w} {np.median(v):.4f}" + (f"(n{len(v)})" if len(v) > 1 else "") for w, v in fam.items()))
good = sum(m <= 0.99 for m in med.values()); bad = [K for K, m in med.items() if m > 1.005]
v = "PASS (0.8B KL test)" if good >= 2 and not bad else "FAIL"
print(f"\nRule 95-01: median <= 0.99 at {good} K ({', '.join(f'K{K:g}' for K, m in med.items() if m <= 0.99) or '-'}), "
      f"> 1.005 at {len(bad)} ({', '.join(f'K{K:g}' for K in bad) or '-'}) -> {v}"
      + ("" if complete and len(med) == len(K1) else "  [INCOMPLETE: not every tensor / K yet]"))

print("\n## 95-02 bf16 (row, 128) scales, +0.125 bpw (16 bits / 128 weights) on the tensor\n")
verd = {}
for meth in ("rs128", "rs128j"):
    print(f"### {meth}" + (" (rule)" if meth == "rs128" else " (joint solve, reported only)"))
    for K in (4, 5, 6):
        for w, ls in T2.items():
            gross, eq, q = [], [], []
            for l in ls:
                rs, b0, b1 = rel(l, w, K, meth), rel(l, w, K), rel(l, w, K + 0.25)
                if rs is None or b0 is None or b1 is None: continue
                gross.append(rs / b0); eq.append(rs / math.sqrt(b0 * b1)); q.append(rs / b1)
            if not eq: print(f"K{K} {w:11s}: no data"); continue
            n = f"n={len(eq)}/{len(ls)}"
            print(f"K{K} {w:11s}: vs K+0.125 (equal bits) median {np.median(eq):.4f} p90 {np.percentile(eq, 90):.4f} | "
                  f"vs K {np.median(gross):.4f} | vs K+0.25 {np.median(q):.4f}  {n}")
            if meth == "rs128" and w != "ffn_down": verd[(K, w)] = (float(np.median(eq)), len(eq) == len(ls))
    print()
ok = [K for K in (5, 6) if all(verd.get((K, w), (9, 0))[0] <= 0.97 for w in ("attn_output", "ssm_out"))]
full = all(verd.get((K, w), (9, False))[1] for K in (5, 6) for w in ("attn_output", "ssm_out"))
print(f"Rule 95-02 (rs128, median vs equal bits <= 0.97 on attn_output and ssm_out at K5 or K6): "
      + ("PASS at " + ", ".join(f"K{K}" for K in ok) + " (0.8B KL test)" if ok else "FAIL")
      + ("" if full else "  [INCOMPLETE]"))
