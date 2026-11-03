#!/usr/bin/env python3
"""selftest_price27.py LEVELS DEMO.jsonl OUT -- fake 27B ladders for price27.sh: ONE/K<k> = the real V3 ladder's
proxy.jsonl (data/v3/levels) with proxy_err1 = proxy_err2 = proxy_err; H/K<k> = the same with proxy_err2 x the demo
ratio (0.8B per-type two-sided ratios, mkratio.py) and proxy_err1 x 1.06 (the one-sided rise seen on the 0.8B), for
K 2.5 and 3; a swaps file with the H level 8% better than ONE at K3."""
import json, os, sys
L, DEMO, O = sys.argv[1:4]
r = {(d["key"], d["K"]): d["r"] for d in map(json.loads, open(DEMO))}
for K in ("2.5", "3"):
    for arm in ("ONE", "H"):
        os.makedirs(f"{O}/{arm}/K{K}", exist_ok=True)
        with open(f"{O}/{arm}/K{K}/proxy.jsonl", "w") as f:
            for l in open(f"{L}/K{K}/proxy.jsonl"):
                d = json.loads(l); e = d["proxy_err"]; d["proxy_err1"] = d["proxy_err2"] = e
                if arm == "H": d["proxy_err1"] = e * 1.06; d["proxy_err2"] = e * r.get((d["key"], float(K)), 1.0)
                f.write(json.dumps(d) + "\n")
open(f"{O}/swaps.txt", "w").write("3 0.0300 0.0276\n")
print("fake ladders written")
