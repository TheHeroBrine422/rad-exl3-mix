#!/usr/bin/env python3
import json, os, sys, numpy as np
# (report only, post hoc) per-document share of s4-3 KL on the search / qad dump rows; ran on Thor
sys.argv=[sys.argv[0]]; sys.path.insert(0, "/data/work/w96/code/v3/yaqa/w96"); import plans96 as p
D = json.load(open("/data/work/w96/code/v3/yaqa/w94/docs94.json")) if os.path.exists("/data/work/w96/code/v3/yaqa/w94/docs94.json") else None
for nm in ("search", "qad"):
    S = p.Sums(nm); full = p.splans(np.load(f"{p.HSD}/headsens.npz")["fisher_qad_gen"])
    a = S.kl(full["s4-3"]); u = S.kl(p.UNI["u4"]); d, inv = np.unique(S.doc, return_inverse=True)
    sa, su, c = np.bincount(inv, a), np.bincount(inv, u), np.bincount(inv)
    o = np.argsort(-sa); share = np.cumsum(sa[o]) / sa.sum()
    print(nm, "docs", len(d), "top-1/3/5 docs share of s4-3 KL", [round(float(share[i]), 3) for i in (0, 2, 4)],
          "ratio s4-3/u4 without top-3 docs", round(float(sa[o[3:]].sum() / su[o[3:]].sum()), 3))
    print("  top docs (doc, rows, s4-3 share, its s/u)", [(int(d[i]), int(c[i]), round(float(sa[i] / sa.sum()), 3), round(float(sa[i] / su[i]), 2)) for i in o[:5]])
