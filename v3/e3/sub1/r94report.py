#!/usr/bin/env python3
"""r94report.py [i8hi.jsonl] -- #94 numbers and verdicts (README "#94", rules fixed before data).

ratio = rel(mul1_i8) / rel(mul1) per tensor and K (same rows / seed / factors), median and p90 over the 10 tensors.
Rule: an i8 level K7.25 / 7.5 / 7.75 joins the ladder if its median ratio <= 1.10 and its p90 <= 1.20.
Sanity: the K7 and K8 medians must reproduce the GPU smoke (x1.024 K7, x1.347 K8, LOG 2026-11-01) within +-0.05,
else the CPU proxy is not trusted here. Also shown: how often the scale search sits on its top factor (1.15 with
i8hi.sh's E3_FACTORS: then the scale grid, not the codebook, may limit both arms). Default: data/e3/sub1/i8hi.jsonl."""
import json, os, sys
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
path = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/e3/sub1/i8hi.jsonl"
R = {}
for l in open(path):
    try: o = json.loads(l)
    except ValueError: continue
    R[(o["layer"], o["w"], o["method"], float(o["bits"]))] = o
T = [(10, "ssm_qkv"), (10, "ssm_out"), (10, "ffn_gate_up"), (10, "ffn_down"), (30, "ffn_down"), (50, "ssm_qkv"),
     (50, "ffn_gate_up"), (19, "attn_qg"), (19, "attn_output"), (43, "attn_v")]
KS = [6.5, 7.0, 7.25, 7.5, 7.75, 8.0]
GPU = {7.0: 1.024, 8.0: 1.347}
print(f"# #94 report from {os.path.relpath(path, ROOT)} ({len(R)} lines)\n")
print("| K | median i8/mul1 | p90 | min | max | n | rule |")
print("|---|---|---|---|---|---|---|")
med, p90, n = {}, {}, {}
for K in KS:
    r = [R[(l, w, "ldlq_mul1_i8", K)]["rel"] / R[(l, w, "ldlq_mul1", K)]["rel"] for l, w in T
         if (l, w, "ldlq_mul1_i8", K) in R and (l, w, "ldlq_mul1", K) in R]
    n[K] = len(r)
    if not r: print(f"| {K:g} | - | - | - | - | 0 | |"); continue
    med[K], p90[K] = float(np.median(r)), float(np.percentile(r, 90))
    tag = (f"GPU smoke x{GPU[K]:.3f}: {'ok' if abs(med[K] - GPU[K]) <= 0.05 else 'OFF'}" if K in GPU else
           ("joins" if med[K] <= 1.10 and p90[K] <= 1.20 else "no") if K in (7.25, 7.5, 7.75) else "")
    print(f"| {K:g} | {med[K]:.4f} | {p90[K]:.4f} | {min(r):.4f} | {max(r):.4f} | {len(r)} | {tag} |")
print("\nper tensor (i8 / mul1):")
print("tensor            " + " ".join(f"{K:>6g}" for K in KS))
for l, w in T:
    print(f"L{l:<2d} {w:13s} " + " ".join(f"{R[(l, w, 'ldlq_mul1_i8', K)]['rel'] / R[(l, w, 'ldlq_mul1', K)]['rel']:6.3f}"
                                         if (l, w, "ldlq_mul1_i8", K) in R and (l, w, "ldlq_mul1", K) in R else "     -" for K in KS))
top = max(o["scale_f"] for o in R.values())
print(f"\nscale search on its top factor ({top:g}): {sum(o['scale_f'] >= top - 1e-9 for o in R.values())}/{len(R)} lines")
# the edge's cost: fracsweep-hi (mul1, scale search 0.85-1.75, same rows / seed) on the shared tensors
F = {}
for l in open(f"{ROOT}/v3/e3/data/fracsweep-hi.jsonl"):
    o = json.loads(l); F[(o["layer"], o["w"], float(o["bits"]))] = o
cmp_ = [(K, R[(l, w, "ldlq_mul1", K)]["rel"] / F[(l, w, K)]["rel"], F[(l, w, K)]["scale_f"]) for l, w in T for K in KS
        if (l, w, K) in F and (l, w, "ldlq_mul1", K) in R]
for K in KS:
    v = [x for k, x, _ in cmp_ if k == K]
    if v: print(f"  mul1 K{K:g}: this run / fracsweep-hi (wider scale grid) median {np.median(v):.3f} n={len(v)}; "
                f"fracsweep-hi's scale factors {sorted({s for k, _, s in cmp_ if k == K})}")
complete = all(n[K] == len(T) for K in KS)
sane = all(K in med and abs(med[K] - GPU[K]) <= 0.05 for K in GPU)
print(f"\nSanity (K7 / K8 within +-0.05 of the GPU smoke): "
      + ", ".join(f"K{K:g} {med[K]:.3f} vs {GPU[K]:.3f}" for K in GPU if K in med) + (" -> ok" if sane else " -> FAIL (proxy not trusted)"))
for K in (7.25, 7.5, 7.75):
    if K not in med: print(f"Rule #94 K{K:g}: no data"); continue
    j = med[K] <= 1.10 and p90[K] <= 1.20
    print(f"Rule #94 K{K:g}: median {med[K]:.3f} (<= 1.10), p90 {p90[K]:.3f} (<= 1.20) -> "
          + ("joins the i8 ladder" if j and sane else "joins, but the sanity check failed: not trusted" if j else "does not join"))
if not complete: print("[INCOMPLETE: " + ", ".join(f"K{K:g} n={n[K]}" for K in KS if n[K] < len(T)) + "]")
