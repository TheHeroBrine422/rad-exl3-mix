#!/usr/bin/env python3
"""cmp.py GOLDEN BASE.rows ARM.rows [--tag T] [--base B] -- one #46-03 pair: mean KLD of both over GOLDEN's scored
rows (gen.u8), the paired difference ARM - BASE with its per-document bootstrap SE (2,000 resamples, seed 0: the same
estimator as swap27's paired.py), the ratio, p99 KLD and top-1 agreement of both. One JSON line."""
import json, sys
import numpy as np

a = sys.argv[1:]
def opt(k):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
tag, base = opt("--tag"), opt("--base")
g, pa, pb = a[:3]
m = json.load(open(f"{g}/kld.json"))
A = np.fromfile(pa, np.float32).reshape(-1, 4).astype(np.float64)
B = np.fromfile(pb, np.float32).reshape(-1, 4).astype(np.float64)
gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
assert len(A) == len(B) == len(gen), (len(A), len(B), len(gen))
sa, sb, n, mask = [], [], [], np.zeros(len(gen), bool)
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); mk = gen[s]
    sa.append(A[s, 0][mk].sum()); sb.append(B[s, 0][mk].sum()); n.append(mk.sum()); mask[s] |= mk
sa, sb, n = map(np.array, (sa, sb, n))
ka, kb = sa.sum() / n.sum(), sb.sum() / n.sum()
rng = np.random.default_rng(0); bd, br = [], []
for _ in range(2000):
    i = rng.integers(0, len(n), len(n)); x, y, z = sa[i].sum(), sb[i].sum(), n[i].sum()
    bd.append((y - x) / z); br.append(y / x)
se = float(np.std(bd))
print(json.dumps({"tag": tag, "base": base, "kld_a": ka, "kld_b": kb, "diff": kb - ka, "se": se,
                  "z": (kb - ka) / se if se > 0 else 0.0, "ratio": kb / ka, "ratio_se": float(np.std(br)),
                  "p99_a": float(np.percentile(A[mask, 0], 99)), "p99_b": float(np.percentile(B[mask, 0], 99)),
                  "top1_a": float(A[mask, 3].mean()), "top1_b": float(B[mask, 3].mean()),
                  "positions": int(n.sum()), "docs": len(n)}))
