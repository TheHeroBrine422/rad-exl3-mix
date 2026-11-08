#!/usr/bin/env python3
"""pwreport82.py PWDIR [SWJ_FITDIR] -- #118 2(a) readout (../w82.md): per arm and part the geometric mean over layers of
rho1 = e1(swj) / e1(plain) (one-sided), rho2 = e2(swj) / e2(plain) (Σdy²-weighted), and the LS-refit's e1 / plain; per K
the geometric mean over arms; next to swj's own MLP-output error ratio from the fit (swiglu.jsonl heldout_joint /
heldout_base, if SWJ_FITDIR is given)."""
import collections, glob, json, math, os, sys
import numpy as np

D = sys.argv[1]; F = sys.argv[2] if len(sys.argv) > 2 else None
KOF = {"base": 2.5, "s1": 2.5, "bn": 2.5, "bns1": 2.5, "base4": 4, "bn4": 4}
g = lambda v: math.exp(np.mean(np.log(v))) if len(v) else float("nan")
byK = collections.defaultdict(lambda: collections.defaultdict(list))
print("arm | K | layers | gate rho1 | up rho1 | gate+up rho1 | gate rho2 | up rho2 | gate+up rho2 | ls/plain e1 | k_mean gate / up | swj MLP-out heldout ratio")
for arm in KOF:
    fs = sorted(glob.glob(f"{D}/{arm}/L*.json"))
    if not fs: continue
    r = collections.defaultdict(list); km = collections.defaultdict(list)
    for f in fs:
        d = json.load(open(f))
        for p in ("gate_proj", "up_proj"):
            x = d[p]
            r[f"{p}1"].append(x["swj"]["e1"] / x["plain"]["e1"]); r[f"{p}2"].append(x["swj"]["e2"] / x["plain"]["e2"])
            r["ls"].append(x["ls"]["e1"] / x["plain"]["e1"]); km[p].append(x["k_mean"])
    both1 = r["gate_proj1"] + r["up_proj1"]; both2 = r["gate_proj2"] + r["up_proj2"]
    mo = ""
    if F and os.path.exists(f"{F}/{arm}/swiglu.jsonl"):
        lines = [json.loads(l) for l in open(f"{F}/{arm}/swiglu.jsonl")]
        mo = f"{g([l['heldout_joint'] / l['heldout_base'] for l in lines]):.3f}"
    K = KOF[arm]
    for k, v in (("1", both1), ("2", both2)): byK[K][k] += v
    byK[K]["mo"] += [float(mo)] if mo else []
    print(f"{arm} | {K:g} | {len(fs)} | {g(r['gate_proj1']):.3f} | {g(r['up_proj1']):.3f} | {g(both1):.3f} | "
          f"{g(r['gate_proj2']):.3f} | {g(r['up_proj2']):.3f} | {g(both2):.3f} | {g(r['ls']):.5f} | "
          f"{np.mean(km['gate_proj']):.3f} / {np.mean(km['up_proj']):.3f} | {mo}")
print("\nK | geo rho1 (gate+up, all arms) | geo rho2 | correction on gate/up one-sided error = 1 / rho1")
for K, v in sorted(byK.items()):
    print(f"{K:g} | {g(v['1']):.3f} | {g(v['2']):.3f} | {1 / g(v['1']):.3f}")
