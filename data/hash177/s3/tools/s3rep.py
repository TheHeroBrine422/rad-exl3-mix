#!/usr/bin/env python3
"""s3rep.py S3DIR GOLDENS -- RULES-177 stage-3 readout: per (K, golden) R = mean KLD(hash) / mean KLD(i8) over the golden's
scored rows (gen.u8), paired per-document bootstrap 95 % CI (2,000 resamples, seed 0: paired.py's resampling, ratio of
sums); the arm-I slope s = ln(KLD(2.625) / KLD(4.125)) / 1.5 per bit on the same golden and Δbpw = ln R / s at each K
(positive = the hash needs that many more bits for equal KLD); labels per RULES-177 (on the deciding golden
golden-search27c-c: "at par" CI contains 1 and R <= 1.01 at both K, "costs" CI above 1, "better" CI below). (w139, #177)"""
import json, math, os, sys
import numpy as np

S3, GD = sys.argv[1], sys.argv[2]
KS = ["2.625", "4.125"]; GOLD = ["golden-search27c-c", "golden-ver3-8k"]


def docsums(g, rows):
    m = json.load(open(f"{GD}/{g}/kld.json")); gen = np.fromfile(f"{GD}/{g}/gen.u8", np.uint8).astype(bool)
    A = np.fromfile(rows, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    s, n = [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); mk = gen[sl]
        s.append(A[sl][mk].sum()); n.append(mk.sum())
    return np.array(s), np.array(n)


res = {}
for g in GOLD:
    for K in KS:
        fi = f"{S3}/q27v3-144dd6e1-k{K}-i8/score-{g}.json.rows"; fh = f"{S3}/q27v3-144dd6e1-k{K}-hash/score-{g}.json.rows"
        if not (os.path.exists(fi) and os.path.exists(fh)): continue
        si, n = docsums(g, fi); sh, n2 = docsums(g, fh); assert (n == n2).all()
        r = sh.sum() / si.sum(); rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(n), len(n)); bs.append(sh[i].sum() / si[i].sum())
        res.setdefault(g, {})[K] = {"kld_i8": si.sum() / n.sum(), "kld_hash": sh.sum() / n.sum(), "ratio": r,
                                    "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                    "positions": int(n.sum()), "docs": len(n)}
for g, d in res.items():
    if all(K in d for K in KS):
        s = math.log(d["2.625"]["kld_i8"] / d["4.125"]["kld_i8"]) / 1.5; d["slope_per_bit"] = s
        for K in KS: d[K]["dbpw"] = math.log(d[K]["ratio"]) / s
    for K in KS:
        if K not in d: continue
        x = d[K]; lo, hi = x["ci"]
        x["label"] = "costs" if lo > 1 else "better" if hi < 1 else "at par" if x["ratio"] <= 1.01 else "inconclusive"
        print(f"{g} K{K}: KLD i8 {x['kld_i8']:.5f} hash {x['kld_hash']:.5f}  R {x['ratio']:.4f} [{lo:.4f}, {hi:.4f}] ({x['label']})"
              + (f"  dbpw {x['dbpw']:+.3f}" if "dbpw" in x else "") + f"  n {x['positions']} / {x['docs']} docs")
d = res.get("golden-search27c-c", {})
if all(K in d for K in KS):
    par = all(d[K]["ci"][0] <= 1 <= d[K]["ci"][1] and d[K]["ratio"] <= 1.01 for K in KS)
    print("RULES-177 label (search27c-c):", "at par" if par else ", ".join(f"K{K} {d[K]['label']}" for K in KS))
if all(K in res.get(g, {}) for g in GOLD for K in KS):
    for K in KS:
        a, b = res[GOLD[0]][K]["ratio"] - 1, res[GOLD[1]][K]["ratio"] - 1
        if a * b < 0 and abs(a) > 0.01 and abs(b) > 0.01: print(f"K{K}: slice-dependent ({a:+.3f} vs {b:+.3f})")
# amendment 1: the fresh xcorp part (env XG = its golden dir) and the pooled two-part ratio (c = search27c-c, x)
XG = os.environ.get("XG")
if XG:
    gx = os.path.basename(XG)
    for K in KS:
        fi = f"{S3}/q27v3-144dd6e1-k{K}-i8/score-{gx}.json.rows"; fh = f"{S3}/q27v3-144dd6e1-k{K}-hash/score-{gx}.json.rows"
        if not (os.path.exists(fi) and os.path.exists(fh)): continue
        GD0 = GD; GD = os.path.dirname(XG); xi, xn = docsums(gx, fi); xh, _ = docsums(gx, fh); GD = GD0
        ci_, cn = docsums("golden-search27c-c", f"{S3}/q27v3-144dd6e1-k{K}-i8/score-golden-search27c-c.json.rows")
        ch, _ = docsums("golden-search27c-c", f"{S3}/q27v3-144dd6e1-k{K}-hash/score-golden-search27c-c.json.rows")
        rx, rc = xh.sum() / xi.sum(), ch.sum() / ci_.sum(); rp = (xh.sum() + ch.sum()) / (xi.sum() + ci_.sum())
        rng = np.random.default_rng(0); bx, bp = [], []
        for _ in range(2000):
            i = rng.integers(0, len(xn), len(xn)); j = rng.integers(0, len(cn), len(cn))
            bx.append(xh[i].sum() / xi[i].sum()); bp.append((xh[i].sum() + ch[j].sum()) / (xi[i].sum() + ci_[j].sum()))
        cx, cp = [float(np.percentile(bx, q)) for q in (2.5, 97.5)], [float(np.percentile(bp, q)) for q in (2.5, 97.5)]
        if cp[0] > 1 and rx > 1 and rc > 1: lab = "costs"
        elif cp[1] < 1 and rx < 1 and rc < 1: lab = "better"
        elif cp[0] <= 1 <= cp[1] and rp <= 1.01: lab = "at par"
        else: lab = "inconclusive"
        res.setdefault("amend1", {})[K] = {"x": {"kld_i8": xi.sum() / xn.sum(), "kld_hash": xh.sum() / xn.sum(), "ratio": rx, "ci": cx,
                                               "positions": int(xn.sum()), "docs": len(xn)}, "c_ratio": rc, "pooled": rp, "pooled_ci": cp, "label": lab}
        print(f"A1 K{K}: x {gx} KLD i8 {xi.sum() / xn.sum():.5f} hash {xh.sum() / xn.sum():.5f}  R_x {rx:.4f} [{cx[0]:.4f}, {cx[1]:.4f}]  "
              f"(c {rc:.4f})  pooled {rp:.4f} [{cp[0]:.4f}, {cp[1]:.4f}] -> {lab}  n_x {int(xn.sum())} / {len(xn)} pieces")
json.dump(res, open(f"{S3}/s3rep.json", "w"), indent=1)
