#!/usr/bin/env python3
"""wstar.py O ARM-PREFIX W... -- d1108: the blend weight w* with the lowest mean log paired ratio vs k{3,2}dygm over
both goldens (calib search, ag-) and both K, from O/result.txt's pair lines (seeds averaged). Prints w* (or "none" if
no weight has all four cells)."""
import math, re, sys
o, pre, ws = sys.argv[1], sys.argv[2], sys.argv[3:]
P = {}
for l in open(f"{o}/result.txt"):
    m = re.search(r"pair (\S+) -> (\S+): \"ratio\": ([0-9.]+)", l)
    if m: P[(m[1], m[2])] = float(m[3])
best = None
for w in ws:
    cells = []
    for k in (3, 2):
        for g in ("", "ag-"):
            rs = [P[(f"{g}k{k}dygm{s}", f"{g}k{k}{pre}{w}{s}")] for s in ("", "_s1") if (f"{g}k{k}dygm{s}", f"{g}k{k}{pre}{w}{s}") in P]
            if rs: cells.append(sum(math.log(r) for r in rs) / len(rs))
    if len(cells) == 4:
        m = sum(cells) / 4; print(f"# {pre}{w}: mean log {m:+.4f}", file=sys.stderr)
        if best is None or m < best[1]: best = (w, m)
print(best[0] if best else "none")
