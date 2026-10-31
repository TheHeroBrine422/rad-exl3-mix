#!/usr/bin/env python3
"""paired.py GOLDEN A.f32 B.f32 -- the KLD difference B - A over the rows GOLDEN's gen.u8 marks, and its standard
error by bootstrap over documents (paired: the same documents in every resample)."""
import json, sys
import numpy as np
g, a, b = sys.argv[1:4]
m = json.load(open(f"{g}/kld.json"))
A = np.fromfile(a, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
B = np.fromfile(b, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
sa, sb, n = [], [], []
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); mk = gen[s]
    sa.append(A[s][mk].sum()); sb.append(B[s][mk].sum()); n.append(mk.sum())
sa, sb, n = map(np.array, (sa, sb, n))
ka, kb = sa.sum() / n.sum(), sb.sum() / n.sum()
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    i = rng.integers(0, len(n), len(n)); bs.append((sb[i].sum() - sa[i].sum()) / n[i].sum())
se = np.std(bs)
print(json.dumps({"kld_a": ka, "kld_b": kb, "diff": kb - ka, "rel": (kb - ka) / ka, "se": se, "z": (kb - ka) / se,
                  "positions": int(n.sum()), "docs": len(n)}))
