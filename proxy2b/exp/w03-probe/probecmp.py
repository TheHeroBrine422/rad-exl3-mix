#!/usr/bin/env python3
"""probecmp.py SAMPLED PROBE [N,N,..] -- precision of two YAQA_DIAG collections over the same windows (c18 = sampled labels,
cprobe = the A10 probe): per snapshot n, the split-half agreement from each collection's halves.json (even vs odd
sequences, n/2 each: noise only, no quant noise) and the per-linear cosine of each snapshot with the OTHER collection's
largest snapshot (shared windows, independent labels). Also the probe's effective dose: the sampled sequence count whose
halves median (log-linear interpolation in n of 1 - median) equals the probe's at n."""
import json, math, os, sys
import numpy as np, torch
from safetensors import safe_open
S, Pr = sys.argv[1:3]
NS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "128,256,512,1024").split(",")]
def halves(d, n):
    f = f"{d}/hdiag-snap{n}/halves.json"
    return json.load(open(f)) if os.path.exists(f) else None
def vecs(d, n):
    p = f"{d}/hdiag-snap{n}"
    out = {}
    for f in sorted(os.listdir(p)):
        if f.endswith(".safetensors"):
            with safe_open(f"{p}/{f}", "pt") as h: out[f[:-12]] = h.get_tensor("hout_diag").double()
    return out
cos = lambda u, v: float((u * v).sum() / (u.norm() * v.norm()))
top = max(n for n in NS if halves(S, n) and halves(Pr, n))
ref = {"sampled": vecs(S, top), "probe": vecs(Pr, top)}
hs = {n: halves(S, n)["median"] for n in NS if halves(S, n)}
def eff(m):                                       # sampled n with the same halves median
    pts = sorted((n, math.log(1 - v)) for n, v in hs.items())
    y = math.log(1 - m)
    for (n0, y0), (n1, y1) in zip(pts, pts[1:]):
        if y1 <= y <= y0: return n0 * (n1 / n0) ** ((y - y0) / (y1 - y0))
    (n0, y0), (n1, y1) = (pts[0], pts[1]) if y > pts[0][1] else (pts[-2], pts[-1])     # extrapolate
    return n0 * (n1 / n0) ** ((y - y0) / (y1 - y0))
print(f"{'n':>5s} | sampled halves med / p5 | probe halves med / p5 | probe eff. sampled n | cos(sampled_n, probe_{top}) med | cos(probe_n, sampled_{top}) med")
res = []
for n in NS:
    hS, hP = halves(S, n), halves(Pr, n)
    if not (hS and hP): continue
    vS, vP = vecs(S, n), vecs(Pr, n)
    cSP = np.median([cos(vS[k], ref["probe"][k]) for k in vS]); cPS = np.median([cos(vP[k], ref["sampled"][k]) for k in vP])
    e = eff(hP["median"]); res.append({"n": n, "sampled": [hS["median"], hS["p5"]], "probe": [hP["median"], hP["p5"]],
                                       "probe_eff_n": e, "cos_s_vs_ptop": cSP, "cos_p_vs_stop": cPS})
    print(f"{n:5d} | {hS['median']:.4f} / {hS['p5']:.4f}        | {hP['median']:.4f} / {hP['p5']:.4f}      | {e:8.0f} ({e / n:.2f}x)       "
          f"| {cSP:.4f}                       | {cPS:.4f}")
print("PROBECMP " + json.dumps(res))
