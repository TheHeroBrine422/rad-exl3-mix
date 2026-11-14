#!/usr/bin/env python3
"""sumS.py OUTDIR -- #177 (w139) real-tensor search curve (qS.py / settleS.py jobs: groups md = mad V16 / V18 / B19-16384 /
B20-16384, mx = mad B20-65536 / V20, mb = mul1i8 V16 / B20-16384): per K and arm the geometric mean over tensors of the
output-side rel. error / mul1i8 V16's (same tensor), with the min / max, and the median LDLQ seconds a tensor relative to
the same code's V16 (the gain sweep excluded). Tensors missing an arm are left out of that arm. (w139, #177)"""
import glob, json, math, os, sys
from collections import defaultdict

D = defaultdict(dict)
for f in glob.glob(os.path.join(sys.argv[1], "*-m[bdx].jsonl")):
    j = json.load(open(f)); g = f.rsplit("-", 1)[1][:2]; code = "mul1i8" if g == "mb" else "mad"
    for a, v in j["rel"].items():
        D[(j["layer"], j["w"], int(j["bits"]))][(code, a)] = (v, j["sec_ldlq"][a])
out = {}
for K in (2, 4):
    T = {k: v for k, v in D.items() if k[2] == K and ("mul1i8", "V16") in v}
    arms = sorted({a for v in T.values() for a in v}, key=lambda x: (x[0] != "mul1i8", int(x[1][1:3]), x[1]))
    for arm in arms:
        r = [v[arm][0] / v[("mul1i8", "V16")][0] for v in T.values() if arm in v]
        t = sorted(v[arm][1] / v[(arm[0], "V16")][1] for v in T.values() if arm in v and (arm[0], "V16") in v)
        gm = math.exp(sum(map(math.log, r)) / len(r))
        out[f"K{K} {arm[0]} {arm[1]}"] = {"geo": gm, "min": min(r), "max": max(r), "n": len(r), "time_x": t[len(t) // 2]}
        print(f"K{K} {arm[0]:7s} {arm[1]:10s} n {len(r):2d}  ×{gm:.4f} [{min(r):.3f}, {max(r):.3f}]  time ×{t[len(t) // 2]:.1f}")
json.dump(out, open(os.path.join(sys.argv[1], "..", "sumS.json"), "w"), indent=1)
