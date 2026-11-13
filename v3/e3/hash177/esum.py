#!/usr/bin/env python3
"""esum.py DIR [BASE] -- estep.sh's whole-model timings: per (pack, home) the decode step ms at C 1 / C 8 (each round's
dstep window; median over rounds, min-max in brackets) and the e2e 8k prefill tok/s (median of every request), with
BASE / home ratios (> 1 = the home is faster). Markdown to stdout. (w139, #177)"""
import glob, os, re, statistics as st, sys
from collections import defaultdict

d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "base"
step = defaultdict(list); pre = defaultdict(list)
for f in glob.glob(os.path.join(d, "dstep-r*-*.txt")):
    r, h, p = re.match(r"dstep-r(\d+)-([^-]+)-(.+)\.txt", os.path.basename(f)).groups()
    for l in open(f):
        x = l.split()
        if len(x) >= 5 and x[0] in ("1", "8") and "<-" not in l: step[(p, h, int(x[0]))].append(float(x[4]))
for f in glob.glob(os.path.join(d, "prefill-r*-*.txt")):
    r, h, p = re.match(r"prefill-r(\d+)-([^-]+)-(.+)\.txt", os.path.basename(f)).groups()
    for l in open(f):
        x = re.search(r"= (\d+) tok/s", l)
        if x: pre[(p, h)].append(float(x.group(1)))
packs = sorted({k[0] for k in step}); homes = sorted({k[1] for k in step}, key=lambda h: (h != base, h != "abl", h))
print("| pack | home | C1 ms/step | C8 ms/step | 8k prefill tok/s | C1 vs base | C8 vs base | prefill vs base |")
print("|---|---|---|---|---|---|---|---|")
for p in packs:
    b1, b8, bp = (st.median(step[(p, base, 1)]) if step[(p, base, 1)] else None, st.median(step[(p, base, 8)]) if step[(p, base, 8)] else None,
                  st.median(pre[(p, base)]) if pre[(p, base)] else None)
    for h in homes:
        s1, s8, pp = step[(p, h, 1)], step[(p, h, 8)], pre[(p, h)]
        f = lambda v: f"{st.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]" if v else "-"
        r = lambda b, v, inv: (f"{(b / st.median(v) if inv else st.median(v) / b):.3f}" if (b and v) else "-")
        print(f"| {p} | {h} | {f(s1)} | {f(s8)} | {f(pp)} | {r(b1, s1, True)} | {r(b8, s8, True)} | {r(bp, pp, False)} |")
