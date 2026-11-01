#!/usr/bin/env python3
"""v2board.py RESULTS.json -- EXL3-V2's leaderboard scores next to IQR-V3 and EXL3-V1 at the same size.
Sizes from the board's bytes (bytes x 8 / params). Competitor curves: log KLD interpolated linearly in bpw between
their entries (EXL3-V1 = the better of turboderp's SC / plain at each size); outside a curve's range: no value."""
import json, math, re, sys

d = json.load(open(sys.argv[1])); P = d["params"]
def pts(pred):
    return sorted((e["bytes"] * 8 / P, e["scores"]["kld_mean"]) for e in d["entries"] if pred(e["name"]))
def interp(c, b):
    for (x0, y0), (x1, y1) in zip(c, c[1:]):
        if x0 <= b <= x1:
            return math.exp(math.log(y0) + (b - x0) / (x1 - x0) * (math.log(y1) - math.log(y0)))
    return None
iqr = pts(lambda n: re.fullmatch(r"IQ\d+R(-[SL])?-V3", n) is not None)
v1raw = pts(lambda n: re.fullmatch(r"EXL3-(SC-)?[\d.]+bpw", n) is not None)
v1 = []                                   # lower envelope of SC and plain
for x, y in v1raw:
    if v1 and abs(v1[-1][0] - x) < 0.02: v1[-1] = (x, min(y, v1[-1][1]))
    else: v1.append((x, y))
v1 = [p for i, p in enumerate(v1) if not any(q[0] <= p[0] and q[1] <= p[1] for q in v1[:i])]
print(f"board updated {d['updated']}")
print("| size | bpw (board) | EXL3-V2 KLD | top-1 | IQR-V3 at the same size | ratio | EXL3-V1 at the same size | ratio |")
print("|---|---|---|---|---|---|---|---|")
for e in sorted(d["entries"], key=lambda e: e["bytes"]):
    m = re.fullmatch(r"EXL3-V2-([\d.]+)bpw", e["name"])
    if not m: continue
    b = e["bytes"] * 8 / P; k = e["scores"]["kld_mean"]
    a, v = interp(iqr, b), interp(v1, b)
    f = lambda x: "—" if x is None else (f"{x:.3g}")
    r = lambda x: "" if x is None else f"{k / x - 1:+.0%}"
    print(f"| {m.group(1)} | {b:.3f} | {k:.3g} | {e['scores']['top1']:.1%} | {f(a)} | {r(a)} | {f(v)} | {r(v)} |")
