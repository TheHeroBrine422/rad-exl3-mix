#!/usr/bin/env python3
"""smisum.py RES ROW... -- #175 (w137) pf.sh clk rows: per row the 6 measured 8k prefills' wall (median, s) and the
rocm-smi samples taken while they ran (sclk / mclk MHz, average package power W, junction temp C: mean and range)."""
import re, statistics as st, sys
res = sys.argv[1]
print(f"{'row':10s} {'wall s':>7s} {'tok/s':>6s} {'sclk MHz mean (min-max)':>24s} {'mclk':>5s} {'power W mean (max)':>19s} {'Tj C':>5s}  n")
for r in sys.argv[2:]:
    b = open(f"{res}/bench-clk-{r}.txt").read()
    w = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r"prompt (\d+) tokens, ([\d.]+) s", b)][1:]
    s = open(f"{res}/smi-clk-{r}.txt").read()
    sc = [int(x) for x in re.findall(r"sclk clock level: \d+: \((\d+)Mhz\)", s)]
    mc = [int(x) for x in re.findall(r"mclk clock level: \d+: \((\d+)Mhz\)", s)]
    pw = [float(x) for x in re.findall(r"Power \(W\): ([\d.]+)", s)]
    tj = [float(x) for x in re.findall(r"junction\) \(C\): ([\d.]+)", s)]
    wall = st.median(t for _, t in w)
    print(f"{r:10s} {wall:7.3f} {st.median(n for n, _ in w) / wall:6.0f} {st.mean(sc):10.0f} ({min(sc)}-{max(sc)}) {st.mean(mc):5.0f} "
          f"{st.mean(pw):11.1f} ({max(pw):.0f}) {st.mean(tj):5.1f}  {len(sc)}")
