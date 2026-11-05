#!/usr/bin/env python3
"""r9002report.py [i4.jsonl] -- #90-02 numbers and verdict (README "#90-02", rule fixed before data).

ratio = rel(int4grid) / rel(mul1_i8) per tensor and K (same rows / seed / factors). Per K: the parameter-weighted mean
ratio (weights N x K of each tensor) and the worst tensor. Rule: below the highest K where the weighted mean <= 1.01
and no tensor > 1.03, the i4 ladder replaces the i8 one; between that K and K3 both are built (only if d1106 passes);
above K3 i8 only. The K reported is the top of the run of passing K counted up from K0.125 (a pass above a failing K
is listed but does not extend it). Default input: data/e3/sub1/i4.jsonl."""
import json, os, sys
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
path = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/e3/sub1/i4.jsonl"
R = {}
for l in open(path):
    try: o = json.loads(l)
    except ValueError: continue
    R[(o["layer"], o["w"], o["method"], float(o["bits"]))] = o
T = [(10, "ssm_qkv"), (10, "ssm_out"), (10, "ffn_gate_up"), (10, "ffn_down"), (30, "ffn_down"), (50, "ssm_qkv"),
     (50, "ffn_gate_up"), (19, "attn_qg"), (19, "attn_output"), (43, "attn_v")]
KS = sorted({k for *_, k in R})
print(f"# #90-02 report from {os.path.relpath(path, ROOT)} ({len(R)} lines)\n")
print("| K | weighted mean i4/i8 | median | max (tensor) | tensors > 1.03 | n | pass |")
print("|---|---|---|---|---|---|---|")
res = {}
for K in KS:
    r, wt = {}, {}
    for l, w in T:
        a, b = R.get((l, w, "ldlq_int4grid", K)), R.get((l, w, "ldlq_mul1_i8", K))
        if a and b: r[(l, w)] = a["rel"] / b["rel"]; wt[(l, w)] = a["N"] * a["K"]
    if not r: continue
    wm = sum(r[t] * wt[t] for t in r) / sum(wt.values())
    tmax = max(r, key=r.get); over = [t for t in r if r[t] > 1.03]
    ok = wm <= 1.01 and not over
    res[K] = (ok, len(r) == len(T))
    print(f"| {K:g} | {wm:.4f} | {np.median(list(r.values())):.4f} | {r[tmax]:.4f} (L{tmax[0]} {tmax[1]}) | "
          f"{', '.join(f'L{l} {w}' for l, w in over) or '-'} | {len(r)} | {'yes' if ok else 'no'} |")
print("\nper tensor (i4 / i8):")
print("tensor            " + " ".join(f"{K:>6g}" for K in KS))
for l, w in T:
    v = [R.get((l, w, "ldlq_int4grid", K)), R.get((l, w, "ldlq_mul1_i8", K))]
    print(f"L{l:<2d} {w:13s} " + " ".join(f"{R[(l, w, 'ldlq_int4grid', K)]['rel'] / R[(l, w, 'ldlq_mul1_i8', K)]['rel']:6.3f}"
                                         if (l, w, "ldlq_int4grid", K) in R and (l, w, "ldlq_mul1_i8", K) in R else "     -" for K in KS))
top = None
for K in KS:
    if not res.get(K, (False,))[0]: break
    top = K
late = [K for K in KS if res.get(K, (False,))[0] and (top is None or K > top)]
complete = all(c for _, c in res.values()) and len(res) == len(KS)
print(f"\nRule #90-02: passing run from K{KS[0]:g} up to "
      + (f"K{top:g}" if top is not None else "none")
      + (f" (isolated passes above it: {', '.join(f'K{K:g}' for K in late)})" if late else "")
      + (f" -> i4 ladder replaces i8 at K <= {top:g}; K{top:g} < K <= 3: both (if d1106 passes); above K3: i8 only"
         if top is not None else " -> the i4 ladder replaces nothing")
      + ("" if complete else "  [INCOMPLETE]"))
