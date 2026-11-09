#!/usr/bin/env python3
import json, os, sys, numpy as np
# (report only, post hoc) token-length-weighted held-out ratios and the share without the jpn / th documents; ran on Thor
sys.argv=[sys.argv[0]]; sys.path.insert(0, "/data/work/w96/code/v3/yaqa/w96"); import plans96 as p
L = {int(k): v for k, v in json.load(open("/data/work/w96/doclen96.json")).items()}
src = {x["doc"]: x["source"] for x in json.load(open("/data/work/w96/code/v3/yaqa/w94/docs94.json"))["sets"]["search"]}
S = p.Sums("search"); T = json.load(open(f"{p.O}/tune.json"))["choice"]
d, inv = np.unique(S.doc, return_inverse=True); c = np.bincount(inv); w = np.array([L[int(x)] for x in d]) / c   # per-row weight
cjk = np.array([src[int(x)] in ("fineweb2-hq/jpn_Jpan", "wikipedia/th") for x in d])
rng = np.random.default_rng(0); ix = rng.integers(0, len(d), (2000, len(d)))
P = {a: p.splans(p.arm_fisher(s)) for a, s in (("base", "base"), ("more-x", T["more-x"]))}
out = {}
for k in p.PLANS3:
    u = S.kl(p.UNI["u" + k[1]]); su = np.bincount(inv, u)
    for a in P:
        s = np.bincount(inv, S.kl(P[a][k]))
        r_tok = (w[None] * 0 + (s * w)[ix]).sum(1) / (su * w)[ix].sum(1)
        out[f"{a}:{k}"] = {"s/u row-weighted": float(s.sum() / su.sum()), "s/u token-weighted": float((s * w).sum() / (su * w).sum()),
                           "ci_token": [float(np.percentile(r_tok, 2.5)), float(np.percentile(r_tok, 97.5))],
                           "s/u without jpn+th docs": float(s[~cjk].sum() / su[~cjk].sum())}
    sb, sm = np.bincount(inv, S.kl(P["base"][k])), np.bincount(inv, S.kl(P["more-x"][k]))
    r = (sm * w)[ix].sum(1) / (sb * w)[ix].sum(1) - 1
    out[f"more-x/base:{k} token-weighted"] = [float((sm * w).sum() / (sb * w).sum() - 1), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
print(json.dumps(out, indent=0)); json.dump(out, open(f"{p.O}/tokenweighted.json", "w"), indent=1)
