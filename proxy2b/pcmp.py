#!/usr/bin/env python3
"""pcmp.py GOLDEN A.f32 B.f32 [--boot N] -- two evalq.py --out score files on one golden: gen-masked mean KLD of each,
ratio B/A, and the paired difference B - A with a per-document bootstrap SE (documents resampled, each contributing
its gen rows' KL sum and count), z = diff / SE. Also (appended after the original keys, so older parsers still work):
the tail quantiles KL99 / KL99.9 of each and their ratios B/A with a per-document bootstrap 95% interval
(--qboot resamples, default 300), and top-1 agreement of each with the paired difference and its SE (user,
2026-11-04: mean KLD decides; tails and top-1 are guards). KL99.9 on ~10^5 positions rests on ~10^2: noisy."""
import json, sys
import numpy as np
a = sys.argv[1:]
N = int(a[a.index("--boot") + 1]) if "--boot" in a else 2000
QN = int(a[a.index("--qboot") + 1]) if "--qboot" in a else 300
g, fa, fb = a[:3]
m = json.load(open(f"{g}/kld.json"))
A = np.fromfile(fa, np.float32).reshape(-1, 4); B = np.fromfile(fb, np.float32).reshape(-1, 4)
assert len(A) == len(B) == m["rows"], (len(A), len(B), m["rows"])
gen = A[:, 2] > 0
sa, sb, n, ta, tb, ka, kb = [], [], [], [], [], [], []
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); gg = gen[s]
    sa.append(A[s, 0][gg].sum()); sb.append(B[s, 0][gg].sum()); n.append(gg.sum())
    ta.append(A[s, 1][gg].sum()); tb.append(B[s, 1][gg].sum()); ka.append(A[s, 0][gg]); kb.append(B[s, 0][gg])
sa, sb, n = map(np.asarray, (sa, sb, n))
ma, mb = sa.sum() / n.sum(), sb.sum() / n.sum()
rng = np.random.default_rng(0); idx = rng.integers(0, len(n), (N, len(n)))
d = (sb[idx].sum(1) - sa[idx].sum(1)) / n[idx].sum(1)
se = d.std(); diff = mb - ma
ta, tb = np.asarray(ta), np.asarray(tb)
t1a, t1b = ta.sum() / n.sum(), tb.sum() / n.sum(); td = (tb[idx].sum(1) - ta[idx].sum(1)) / n[idx].sum(1)
KA, KB = np.concatenate(ka), np.concatenate(kb); qs = (0.99, 0.999)
qa, qb = np.quantile(KA, qs), np.quantile(KB, qs)
qr = []
for b in rng.integers(0, len(n), (QN, len(n))):
    xa = np.concatenate([ka[i] for i in b]); xb = np.concatenate([kb[i] for i in b])
    qr.append(np.quantile(xb, qs) / np.quantile(xa, qs))
lo, hi = np.percentile(np.asarray(qr), [2.5, 97.5], axis=0)
print(json.dumps({"A": fa, "B": fb, "kld_A": float(ma), "kld_B": float(mb), "ratio": float(mb / ma),
                  "diff": float(diff), "se": float(se), "z": float(diff / se), "docs": len(n), "positions": int(n.sum()),
                  "kl99_A": float(qa[0]), "kl99_B": float(qb[0]), "kl99_ratio": float(qb[0] / qa[0]), "kl99_ci": [float(lo[0]), float(hi[0])],
                  "kl999_A": float(qa[1]), "kl999_B": float(qb[1]), "kl999_ratio": float(qb[1] / qa[1]), "kl999_ci": [float(lo[1]), float(hi[1])],
                  "top1_A": float(t1a), "top1_B": float(t1b), "top1_diff": float(t1b - t1a), "top1_se": float(td.std())}))
