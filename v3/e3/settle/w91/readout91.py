#!/usr/bin/env python3
"""readout91.py O -- ../w91.md item 2's rule over q91.sh's outputs (O = /data/yb/w91): coherent share c (c_exact, the
finite-seed-corrected KL split) at K2 / K3 from split-k{2,3}.json; the bias oracle's r = KL(bias) / KL(plain) per seed
(s0-s2, gen rows of g08-search1, paired on the same rows); the rule. Writes O/readout91.json."""
import json, math, os, sys
import numpy as np

O = sys.argv[1]
sfx = lambda s: "" if s == 0 else f"_s{s}"
kl = lambda f: np.fromfile(f, np.float32).reshape(-1, 4)
g3 = open(f"{O}/g3.txt").read() if os.path.exists(f"{O}/g3.txt") else ""
out = {"g3_pass": "FAIL" not in g3 and g3.count("PASS") == 12}
for k in (2, 3):
    sp = json.load(open(f"{O}/split-k{k}.json"))["all"]
    rs = []
    for s in (0, 1, 2):
        a, b = kl(f"{O}/k{k}base{sfx(s)}.f32"), kl(f"{O}/kb{k}{sfx(s)}.f32"); g = a[:, 2] > 0
        rs.append(float(b[g, 0].astype(np.float64).sum() / a[g, 0].astype(np.float64).sum()))
    geo = math.exp(sum(map(math.log, rs)) / len(rs))
    out[f"K{k}"] = {"c_exact": sp["c_exact"], "c_ci95": sp["c_exact_ci95"], "c_var": sp["c_var"], "ratio_avg_single": sp["ratio_avg_single"],
                    "kl_single": sp["kl_single"], "kl_avg": sp["kl_avg"], "bias_r": rs, "bias_geo": geo,
                    "bias_pass": geo <= 0.99 and all(r < 1 for r in rs)}
    print(f"K{k}: c_exact {sp['c_exact']:.4f} {[round(x, 4) for x in sp['c_exact_ci95']]} (c_var {sp['c_var']:.4f}; KL_avg/KL_single "
          f"{sp['ratio_avg_single']:.4f}); bias oracle r {[round(r, 4) for r in rs]} geo {geo:.4f}")
c2, c3 = out["K2"]["c_exact"], out["K3"]["c_exact"]
if not out["g3_pass"]: v = "NOT READ (G3 failed)"
elif c2 < 0.10 and c3 < 0.10: v = "NEGATIVE (c < 0.10 at both K)"
elif any(out[f"K{k}"]["c_exact"] >= 0.10 and out[f"K{k}"]["bias_pass"] for k in (2, 3)): v = "POSITIVE"
else: v = "NEGATIVE for a field (coherent but not bias-shaped)"
out["verdict"] = v
json.dump(out, open(f"{O}/readout91.json", "w"), indent=1)
print(f"W91 ITEM 2: {v}")
