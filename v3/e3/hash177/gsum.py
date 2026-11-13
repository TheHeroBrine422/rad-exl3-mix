#!/usr/bin/env python3
"""gsum.py DIR [BASE] -- gbench.sh's per-shape timings summarised: per (rate, M, mode) the mean decoder layer (48 GDN :
16 attention, bench_twopass's own line) and each shape, median over rounds, per build; ratios BASE / build (> 1 = the
build is faster) and the decode share (1 - abl / build) when an `abl` build ran. Markdown to stdout. (w139, #177)"""
import csv, glob, os, re, statistics as st, sys
from collections import defaultdict

d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "base"
lay = defaultdict(list)    # (rate, M, mode, build) -> layer us per round
shp = defaultdict(list)    # (rate, M, mode, build, shape) -> us
for f in glob.glob(os.path.join(d, "r*-k*.txt")):
    m = re.match(r"r(\d+)-(.+)-k([\d.]+)\.txt", os.path.basename(f))
    _, b, k = m.groups()
    for l in open(f):
        x = re.match(r"layer M\s+(\d+) (\S+)\s*:.*mean \(48 GDN : 16 attention\)\s+([\d.]+) us", l)
        if x: lay[(float(k), int(x.group(1)), x.group(2), b)].append(float(x.group(3)))
    c = f[:-4] + ".csv"
    if os.path.exists(c):
        for r in csv.DictReader(open(c)):
            shp[(float(k), int(r["M"]), r["mode"], b, r["shape"])].append(float(r["us_median"]))
builds = sorted({k[3] for k in lay}, key=lambda b: (b != base, b != "abl", b))
others = [b for b in builds if b not in (base, "abl")]
keys = sorted({k[:3] for k in lay})
med = lambda v: st.median(v) if v else float("nan")
print(f"## mean layer us (median of rounds); {base} / build (> 1 = faster than {base}); decode share = 1 - abl / build\n")
print("| K | M | mode | " + " | ".join(builds) + " | " + " | ".join(f"{base}/{b}" for b in others + (["abl"] if "abl" in builds else []))
      + " | decode share " + " / ".join([base] + others) + " | rounds |")
print("|" + "---|" * (4 + len(builds) + len(others) + (1 if "abl" in builds else 0) + 1))
for k in keys:
    t = {b: med(lay[k + (b,)]) for b in builds}
    rat = [t[base] / t[b] for b in others + (["abl"] if "abl" in builds else [])]
    sh = [1 - t["abl"] / t[b] for b in [base] + others] if "abl" in builds else []
    print(f"| {k[0]:g} | {k[1]} | {k[2]} | " + " | ".join(f"{t[b]:.1f}" for b in builds) + " | " +
          " | ".join(f"{r:.3f}" for r in rat) + " | " + " / ".join(f"{s:.0%}" for s in sh) +
          f" | {min(len(lay[k + (b,)]) for b in builds)} |")
shapes = sorted({k[4] for k in shp})
print(f"\n## per shape, {base} / build (us of {base} in brackets)\n")
for k in keys:
    if not any(k + (base, s) in shp for s in shapes): continue
    print(f"- K {k[0]:g} M {k[1]} {k[2]}: " + "; ".join(
        f"{s} ({med(shp[k + (base, s)]):.1f}) " + " ".join(f"{b} {med(shp[k + (base, s)]) / med(shp[k + (b, s)]):.3f}"
                                                         for b in others + (["abl"] if "abl" in builds else []))
        for s in shapes if shp.get(k + (base, s))))
