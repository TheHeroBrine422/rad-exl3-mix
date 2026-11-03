#!/usr/bin/env python3
"""summ.py O BASE ARM [BASE ARM ...] -- per-seed paired ratios of ARM vs BASE from O/result.txt's "pair" lines
(seed suffixes "", _s1 .. _s5), their mean, the mean log ratio and its SE over seeds, and whether every seed is < 1."""
import math, re, sys
o, args = sys.argv[1], sys.argv[2:]
P = {}
for l in open(f"{o}/result.txt"):
    m = re.search(r"pair (\S+) -> (\S+): \"ratio\": ([0-9.]+).*\"z\": ([0-9.e+-]+)", l)
    if m: P[(m[1], m[2])] = (float(m[3]), float(m[4]))
for a, b in zip(args[::2], args[1::2]):
    rs = []
    for s in ["", "_s1", "_s2", "_s3", "_s4", "_s5"]:
        if (a + s, b + s) in P: rs.append((s or "_s0", *P[(a + s, b + s)]))
    if not rs: print(f"{b} vs {a}: no pairs"); continue
    lg = [math.log(r) for _, r, _ in rs]; n = len(lg); m = sum(lg) / n
    se = (sum((x - m) ** 2 for x in lg) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else float("nan")
    print(f"{b} vs {a}: n {n} | " + " ".join(f"{s[1:]} {r:.4f} (z {z:+.1f})" for s, r, z in rs)
          + f" | mean ratio {sum(r for _, r, _ in rs) / n:.4f}, geo {math.exp(m):.4f}, mean log {m:+.4f} ± {se:.4f} SE"
          + f" | every seed < 1: {all(r < 1 for _, r, _ in rs)}")
