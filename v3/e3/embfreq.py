#!/usr/bin/env python3
"""embfreq.py ROWSDIR COUNTS.npy -- the embedding menu sweep's per-row errors (run3.sh, rows3/r-K<k>-G<g>.npz) reweighted
by token frequency (search + qad pool counts): does the row-scale vs group choice, or the error curve, change when
frequent tokens count more? Weights: uniform (the sweep), frequency, sqrt(frequency). Error = Σ w err / Σ w energy.
Also the share of the 4096 sampled rows that are never seen in the counts."""
import glob, math, re, sys
import numpy as np
D, C = sys.argv[1:3]; cnt = np.load(C).astype(np.float64)
R = {}
for p in glob.glob(f"{D}/r-K*-G*.npz"):
    k, g = re.findall(r"K([\d.]+)-G(\d+)", p)[0]; z = np.load(p); R[(float(k), int(g))] = z
rows = next(iter(R.values()))["rows"]; c = cnt[rows]
print(f"{len(rows)} sampled rows: {np.mean(c == 0):.1%} never seen; the seen ones carry {c.sum():.0f} of {cnt.sum():.0f} tokens")
W = {"uniform": np.ones_like(c), "freq": c, "sqrt": np.sqrt(c)}
def rel(z, w): return float((w * z["err"]).sum() / (w * z["energy"]).sum())
Ks = sorted({k for k, _ in R})
for name, w in W.items():
    row = [(k + 16 / 5120, math.log2(rel(R[(k, 5120)], w))) for k in Ks]
    def at(b):
        for (b0, e0), (b1, e1) in zip(row, row[1:]):
            if b0 <= b <= b1: return e0 + (b - b0) / (b1 - b0) * (e1 - e0)
    out = []
    for k in (2.0, 3.0, 4.0, 5.0, 5.5):
        s = f"K{k:g}: rel {rel(R[(k, 5120)], w):.5f}"
        if (k, 1024) in R and at(k + 16 / 1024) is not None:
            s += f", G1024 x{2 ** (math.log2(rel(R[(k, 1024)], w)) - at(k + 16 / 1024)):.3f}"
        out.append(s)
    step = [rel(R[(Ks[i], 5120)], w) / rel(R[(Ks[i + 1], 5120)], w) for i in range(len(Ks) - 1)]
    print(f"{name:8s} " + " | ".join(out) + f" | per quarter step x{np.median(step):.2f}")
