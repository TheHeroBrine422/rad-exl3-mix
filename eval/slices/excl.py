#!/usr/bin/env python3
"""excl.py LEDGER.jsonl POOLS.json OUTDIR MODE [GOLDEN_OR_KLD ...] -- the calib exclusion list for a fresh search slice
(w120, 11-12): OUTDIR/kld.json = {"docs": [{"orig": i}, ...]} (calib_slice.py's EXCLUDE_GOLDEN form) holding every
calib document that is NOT fresh, plus the docs[].orig of any extra calib golden given (slices built earlier in the same
run, before their ledger rows exist).
MODE never  fresh = in no ledger row at all (whole-calib "doc": "all" collections do not count; the ledger's rule)
     no27b  fresh = in no ledger row except the 0.8B / 2B proxy slices (uses g08-*, g2b-*): those documents were scored
            by other models only. Needs the lead's OK before a deciding slice uses it (README: calib budget).
Prints how many search-pool documents stay fresh."""
import json, os, sys

led, pools, od, mode = sys.argv[1:5]; extra = sys.argv[5:]
assert mode in ("never", "no27b"), mode
pool = json.load(open(pools))["pool"]
used = set()
for l in open(led):
    r = json.loads(l)
    if r["doc"] == "all": continue
    if mode == "no27b" and r["use"].startswith(("g08-", "g2b-")): continue
    used.add(int(r["doc"]))
for g in extra:
    p = g if g.endswith(".json") else f"{g}/kld.json"
    used |= {int(d["orig"]) for d in json.load(open(p))["docs"]}
os.makedirs(od, exist_ok=True)
json.dump({"docs": [{"orig": i} for i in sorted(used)], "mode": mode}, open(f"{od}/kld.json", "w"))
fresh = [i for i, p in enumerate(pool) if p == "search" and i not in used]
print(f"excl {mode}: {len(used)} calib docs excluded; {len(fresh)} search-pool docs fresh")
