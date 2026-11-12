#!/usr/bin/env python3
"""summary119.py REPDIR -- w119 pricing summary from report119's outputs (no new data read).
  1. a8 head cost per uniform donor as bits: dbits = log2(KL_a8 / KL_f16) / slope, slope = log2 KL per bit between the
     neighbouring uniform donors of the same family (search-gen, calib-weighted; e2.json + e9-search).
  2. deployable head menu today (uniform R16 sigma_out 0.03 i8 K3-6.5, K7 mul1, deadcode's int8 GPTQ, bf16): fit-gen
     KL, its lower hull, and the head's marginal KL per model bpw between neighbours vs the trunk's (knap smoke
     2026-11-12: 6.0 / 6.5 / 7.0 bpw pred 7.51e-4 / 5.62e-4 / 4.96e-4). Model bpw per head bit = 1.271e9 / 27e9."""
import json, math, sys

R = sys.argv[1]
e2 = json.load(open(f"{R}/e2.json")); se = {o["opt"]: o for o in json.load(open(f"{R}/e9-search-T03.json"))}
fi = {o["opt"]: o for o in json.load(open(f"{R}/e9-fit-T03.json"))}
print("a8 head cost (search-gen): option, a8/f16, abs dKL, bit-equivalent")
for o, v in sorted(e2["a8"].items()):
    if "-K" not in o: print(f"  {o:8s} {v['ratio'][0]:.3f}  {v['dkl']:+.2e}"); continue
    fam, k = o.split("-K"); k = float(k)
    nb = [x for x in (k - 1, k + 1) if f"{fam}-K{x:g}" in se]
    sl = sum(abs(math.log2(se[f"{fam}-K{x:g}"]["kl"] / se[o]["kl"])) / abs(x - k) for x in nb) / len(nb)
    print(f"  {o:8s} {v['ratio'][0]:.3f}  {v['dkl']:+.2e}  {math.log2(v['ratio'][0]) / sl:.3f} bits (slope {sl:.2f})")
menu = ["R16s003-K3", "R16s003-K4", "R16s003-K5", "R16s003-K6", "R16s003-K6.5", "R16s003-K7", "R16s003m1-K7", "i8-gptq", "bf16-full"]
pts = sorted((fi[m]["bits"], fi[m]["kl"], m) for m in menu)
hull = []
for p in pts:
    while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0: hull.pop()
    hull.append(p)
PB = 1.271e9 / 27e9
print("\ndeployable head menu (fit-gen calib-weighted head-only KL; hull = lower convex hull of this menu)")
for b, v, m in pts: print(f"  {m:14s} {b:6.3f} b  {v:.3e}  {'HULL' if (b, v, m) in hull else ''}")
print(f"\nhead marginal KL per model bpw along the hull (1 head bit = {PB:.4f} bpw); trunk 6.0-6.5: 3.8e-4, 6.5-7.0: 1.3e-4")
for (b0, v0, m0), (b1, v1, m1) in zip(hull, hull[1:]):
    print(f"  {m0} -> {m1}: {(v0 - v1) / ((b1 - b0) * PB):.2e} per bpw")
