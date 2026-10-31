#!/usr/bin/env python3
"""fracreport.py SWEEP.jsonl -- fractional-K trellis vs interpolation (fracsweep). For every tensor and fractional K,
the measured relative proxy error against the log-linear interpolation between its integer neighbours (K0 = dropped
tensor, rel = 1), same rows at every K: ratio > 1 = the fractional pattern loses against the curve. Also the
measured slope (log2 error ratio per bit) per integer step."""
import json, sys, collections, math, numpy as np

R = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    d = json.loads(l); R[(d["layer"], d["w"])][d["bits"]] = d["rel"]
fr = sorted({k for r in R.values() for k in r if k != int(k)})
print(f"{'tensor':22s}" + "".join(f"{k:>7}" for k in fr) + "   slope/bit K1→2  K2→3  K3→4  K4→5")
ratios = collections.defaultdict(list); slopes = collections.defaultdict(list)
for (l, w), r in sorted(R.items()):
    r = dict(r); r[0.0] = 1.0
    row = []
    for k in fr:
        lo, hi = math.floor(k), math.ceil(k)
        if k in r and lo in r and hi in r:
            t = k - lo; pred = math.exp((1 - t) * math.log(r[lo]) + t * math.log(r[hi]))
            q = r[k] / pred; ratios[k].append(q); row.append(f"{q:7.3f}")
        else: row.append("      -")
    sl = []
    for a in (1, 2, 3, 4):
        if a in r and a + 1 in r: s = math.log2(r[a] / r[a + 1]); slopes[a].append(s); sl.append(f"{s:5.2f}")
        else: sl.append("    -")
    print(f"{l:2d} {w:19s}" + "".join(row) + "   " + "  ".join(sl))
print(f"{'median':22s}" + "".join(f"{np.median(ratios[k]):7.3f}" for k in fr) + "   " +
      "  ".join(f"{np.median(slopes[a]):5.2f}" for a in (1, 2, 3, 4)))
print("ratio = measured / log-linear interpolation between the integer neighbours (1.000 = on the curve)")
