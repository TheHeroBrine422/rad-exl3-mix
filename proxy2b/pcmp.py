#!/usr/bin/env python3
"""pcmp.py GOLDEN A.f32 B.f32 [--boot N] -- two evalq.py --out score files on one golden: gen-masked mean KLD of each,
ratio B/A, and the paired difference B - A with a per-document bootstrap SE (documents resampled, each contributing
its gen rows' KL sum and count), z = diff / SE."""
import json, sys
import numpy as np
a = sys.argv[1:]
N = int(a[a.index("--boot") + 1]) if "--boot" in a else 2000
g, fa, fb = a[:3]
m = json.load(open(f"{g}/kld.json"))
A = np.fromfile(fa, np.float32).reshape(-1, 4); B = np.fromfile(fb, np.float32).reshape(-1, 4)
assert len(A) == len(B) == m["rows"], (len(A), len(B), m["rows"])
gen = A[:, 2] > 0
sa, sb, n = [], [], []
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); gg = gen[s]
    sa.append(A[s, 0][gg].sum()); sb.append(B[s, 0][gg].sum()); n.append(gg.sum())
sa, sb, n = map(np.asarray, (sa, sb, n))
ma, mb = sa.sum() / n.sum(), sb.sum() / n.sum()
rng = np.random.default_rng(0); idx = rng.integers(0, len(n), (N, len(n)))
d = (sb[idx].sum(1) - sa[idx].sum(1)) / n[idx].sum(1)
se = d.std(); diff = mb - ma
print(json.dumps({"A": fa, "B": fb, "kld_A": float(ma), "kld_B": float(mb), "ratio": float(mb / ma),
                  "diff": float(diff), "se": float(se), "z": float(diff / se), "docs": len(n), "positions": int(n.sum())}))
