#!/usr/bin/env python3
"""cmp115.py GOLDEN A_ROWS B_ROWS -- two per-row KLD files over GOLDEN (both [rows, 4] f32, KL in column 0: stock
radiance --kld-out FILE.rows and our recorder's score.f32), gen.u8-masked means and their paired difference B - A with a
per-document bootstrap SE."""
import json, sys
import numpy as np
g, a, b = sys.argv[1:4]
m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
A = np.fromfile(a, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
B = np.fromfile(b, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
assert len(A) == len(B) == m["rows"], (len(A), len(B), m["rows"])
sa, sb, n = [], [], []
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); gg = gen[sl]
    sa.append(A[sl][gg].sum()); sb.append(B[sl][gg].sum()); n.append(gg.sum())
sa, sb, n = map(np.array, (sa, sb, n))
rng = np.random.default_rng(0); bs = rng.integers(0, len(n), (2000, len(n)))
rel = sb[bs].sum(1) / sa[bs].sum(1) - 1
print(json.dumps({"mean_a": sa.sum() / n.sum(), "mean_b": sb.sum() / n.sum(), "rel_b_vs_a": sb.sum() / sa.sum() - 1,
                  "se": float(rel.std()), "rows_scored": int(n.sum()), "max_abs_row_diff": float(np.abs(A[gen] - B[gen]).max())}))
