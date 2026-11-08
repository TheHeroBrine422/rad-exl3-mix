#!/usr/bin/env python3
"""pick91.py CENSUS.jsonl -- ../w91.md item 1's patch-set rule, applied to census91.py's output. Writes CENSUS.pick.json:
  S      = {3994} + every row holding >= 10% of e or of o in any L0-L3 writer, or >= 1/2 of row 3994's e-share in that
           writer; at most 8 rows, ranked by their largest e-share (3994 always in)
  keys8  = the 8 writers of L0-L3
  D      = whether the secondary depth arm runs (row 3994 >= 10% of e in >= 4 of the 120 writers of L4-L63) and its keys
           (every writer of L0-L63 with row-3994 e-share >= 10%)"""
import json, os, sys

import numpy as np

cen = sys.argv[1]
R = {json.loads(x)["key"]: json.loads(x) for x in open(cen) if x.strip()}
early = [k for k, r in R.items() if r["layer"] <= 3]
assert len(early) == 8, f"census has {len(early)} L0-L3 writers, need 8"
cand = {3994: 1.0}
for k in early:
    s = np.load(os.path.join(cen + ".d", k + ".npy")); es, os_ = s[0], s[1]
    hit = np.nonzero((es >= 0.10) | (os_ >= 0.10) | (es >= 0.5 * es[3994]))[0]
    for r in hit.tolist():
        if r != 3994: cand[r] = max(cand.get(r, 0.0), float(es[r]))
S = [3994] + sorted([r for r in cand if r != 3994], key=lambda r: -cand[r])[:7]
deep = [k for k, r in R.items() if r["layer"] >= 4]
deep_hit = [k for k in deep if R[k]["e3994"] >= 0.10]
runD = len(deep) == 120 and len(deep_hit) >= 4
out = {"S": sorted(S), "candidates": {str(r): v for r, v in cand.items()},
       "keys8": sorted(early, key=lambda k: (R[k]["layer"], k)),
       "deep_tensors": len(deep), "deep_e3994_ge_0.10": deep_hit, "runD": runD,
       "keysD": sorted([k for k, r in R.items() if r["e3994"] >= 0.10], key=lambda k: (R[k]["layer"], k)) if runD else []}
json.dump(out, open(cen + ".pick.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("keysD",)}))
