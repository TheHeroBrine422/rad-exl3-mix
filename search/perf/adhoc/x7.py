# x7.py -- heavy documents: largest document's share of A's KL, and the 20% water-filled readout (readout.py) vs pooled
import json, numpy as np, collections
R = [json.loads(l) for l in open("se2/pairs.jsonl")]
K = collections.defaultdict(list)
for r in R:
    if not 0.005 <= abs(r["logr"]) <= 0.15: continue
    k = "x" if r["part"].endswith("-x") else "c" if r["part"].endswith("-c") else r["part"]
    K[k].append(r)
print("part kind        pairs  maxshare med  P(>0.2)  SE cap20/SE med  |cap20-pooled|/SE med  sign flips")
for k, v in sorted(K.items()):
    ms = np.array([r["max_share"] for r in v]); se = np.array([r["se_boot"] for r in v]); sc = np.array([r["se_cap20"] for r in v])
    d = np.array([abs(r["logr_cap20"] - r["logr"]) for r in v]); fl = np.mean([np.sign(r["logr_cap20"]) != np.sign(r["logr"]) for r in v])
    print(f"{k:16s} {len(v):5d}  {np.median(ms):11.2f}  {np.mean(ms > 0.2):7.2f}  {np.median(sc / se):15.2f}  {np.median(d / se):21.2f}  {fl:10.2f}")
