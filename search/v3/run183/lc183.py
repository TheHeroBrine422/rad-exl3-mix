#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""lc183.py GOLDEN NAME=ROWS ... -- #183 long-context gap: gen-row KL per position bin (0-1k, 1-2k, 2-4k, 4-8k) per
arm, the documents carrying the excess of the first arm over the second, and those documents' position bins."""
import json, sys
import numpy as np
G = sys.argv[1]; F = dict(a.split("=", 1) for a in sys.argv[2:])
m = json.load(open(G + "/kld.json")); gen = np.fromfile(G + "/gen.u8", np.uint8).astype(bool); N = m["rows"]
pos = np.zeros(N, int); doc = np.zeros(N, int)
for j, d in enumerate(m["docs"]):
    n = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + n); pos[s] = d["score_from"] + np.arange(n); doc[s] = j
X = {k: np.fromfile(f, np.float32).reshape(N, -1)[:, 0].astype(float) for k, f in F.items()}
E = [0, 1024, 2048, 4096, 8192]; b = np.digitize(pos, E) - 1
mean = lambda x, s: x[s].mean() if s.any() else float("nan")
print(f"docs {len(m['docs'])}, gen rows {gen.sum()}")
print("bin          rows  " + "  ".join(f"{k:>9}" for k in X))
for k in range(4):
    s = gen & (b == k); print(f"{E[k]:>5}-{E[k+1]:<5} {s.sum():6d}  " + "  ".join(f"{mean(X[a], s):9.4f}" for a in X))
print(f"all         {gen.sum():6d}  " + "  ".join(f"{mean(X[a], gen):9.4f}" for a in X))
a0, a1 = list(X)[:2]; o, q = X[a0], X[a1]
ds = sorted(((j, o[gen & (doc == j)].sum(), q[gen & (doc == j)].sum(), int((gen & (doc == j)).sum()), m["docs"][j]["tokens"])
             for j in range(len(m["docs"]))), key=lambda r: -(r[1] - r[2]))
ex = o[gen].sum() - q[gen].sum(); cum = 0
print(f"\nKL sum {a0} {o[gen].sum():.0f}, {a1} {q[gen].sum():.0f}, excess {ex:.0f}; top docs by excess:")
for r in ds[:8]:
    cum += r[1] - r[2]; print(f"  doc {r[0]:3d}: {a0} {r[1]:7.1f}  {a1} {r[2]:7.1f}  gen rows {r[3]:5d}  tokens {r[4]:5d}  cum share {cum / ex:.0%}")
top = [r[0] for r in ds[:3]]; keep = gen & ~np.isin(doc, top)
print("without the top-3 docs: " + "  ".join(f"{a} {mean(X[a], keep):.4f}" for a in X))
for j in top:
    s = gen & (doc == j)
    print(f"doc {j}: " + " | ".join(f"{E[k]}-{E[k+1]} {a0} {mean(o, s & (b == k)):.3f} {a1} {mean(q, s & (b == k)):.3f}" for k in range(4)))
