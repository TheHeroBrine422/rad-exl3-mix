#!/usr/bin/env python3
"""docstat.py GOLD_X GOLD_C A_X.rows A_C.rows B_X.rows B_C.rows -- per-document view of a paired A -> B comparison on a
two-part slice: pooled ratio, per-document geo / median, each document's share of A's KL, the ratio without documents
holding > 20% of a part's KL (the readout rule in TODO, 11-09), leave-one-out range, and the ratio as the k heaviest
documents are dropped. Document bootstrap (2000, seed 0) for the pooled and trimmed ratios."""
import json, sys
import numpy as np
def docs(g, a, b):
    m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
    A = np.fromfile(a, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    B = np.fromfile(b, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    out = []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); mk = gen[s]
        out.append((A[s][mk].sum(), B[s][mk].sum(), mk.sum(), str(d.get("source", d.get("name", "")))[:40]))
    return out
def boot(sa, sb, keep=None, n=2000):
    rng = np.random.default_rng(0); r = []
    for _ in range(n):
        i = rng.integers(0, len(sa), len(sa)); r.append(sb[i].sum() / sa[i].sum())
    return np.percentile(r, [2.5, 97.5]), np.std(r)
gx, gc, ax, ac, bx, bc = sys.argv[1:7]
for name, rows in (("xcorp", docs(gx, ax, bx)), ("calib", docs(gc, ac, bc)), ("both", docs(gx, ax, bx) + docs(gc, ac, bc))):
    sa = np.array([r[0] for r in rows]); sb = np.array([r[1] for r in rows]); src = [r[3] for r in rows]
    ok = sa > 0; rr = sb[ok] / sa[ok]; share = sa / sa.sum()
    ci, se = boot(sa, sb); R = sb.sum() / sa.sum()
    print(f"== {name}: {len(sa)} docs  pooled r {R:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] z {(R-1)/se:+.2f}  "
          f"per-doc geo {np.exp(np.log(rr).mean()):.4f}  median {np.median(rr):.4f}  docs better {np.mean(rr<1):.2f}")
    o = np.argsort(-share)
    print("   heaviest: " + "; ".join(f"{share[i]*100:.1f}% r {sb[i]/sa[i]:.3f} {src[i]}" for i in o[:4]))
    big = share > 0.20
    if big.any() and name != "both":
        k = ~big; ci2, se2 = boot(sa[k], sb[k]); R2 = sb[k].sum() / sa[k].sum()
        print(f"   without {big.sum()} doc(s) > 20% of the part: r {R2:.4f} [{ci2[0]:.4f}, {ci2[1]:.4f}] z {(R2-1)/se2:+.2f}")
    loo = [(sb.sum() - sb[i]) / (sa.sum() - sa[i]) for i in range(len(sa))]
    print(f"   leave-one-out r range {min(loo):.4f} .. {max(loo):.4f}")
    print("   drop k heaviest: " + "  ".join(f"k{k} {sb[o[k:]].sum()/sa[o[k:]].sum():.4f}" for k in (0, 1, 2, 3, 5, 10)))
