#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ledger226.py PLAN.json CALIB_KLD.json -- append the deep-context plan's calib documents (plan226-build.py, the w226 block) to eval/pools/ledger.jsonl as
use qad226-deep, role train (whole documents: from 0 to their length). Skips (doc, use) pairs already there; checks every doc is fit / search (pools-v3)."""
import json, os, sys
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
plan, kld = sys.argv[1:3]; LED = os.path.join(REPO, "eval/pools/ledger.jsonl")
docs = json.load(open(kld))["docs"]; pool = json.load(open(os.path.join(REPO, "eval/pools/pools-v3.json")))["pool"]
have = {(r["doc"], r["use"]) for r in map(json.loads, open(LED))}
ids = [int(k.split(":")[1]) for k in json.load(open(plan))["w226"]["deep_tokens"] if k.startswith("calib:")]
NOTE = ("#226 (w226, 2026-11-23): deep-context QAD plan (v3/qad226/plan226-build.py): the calib fit / search docs longer than 8,192 tokens that are not verify, "
        "not golden-jlong, not j217, not train214, not a QAD monitor slice (966 left out), trained whole (--cap 262144); the 24 longest first in their streams "
        "(front_items), w222's extended plan after them; D256-25 / D256-30")
n = 0
with open(LED, "a") as f:
    for i in ids:
        assert pool[i] in ("fit", "search"), (i, pool[i])
        if (i, "qad226-deep") in have: continue
        f.write(json.dumps({"doc": i, "use": "qad226-deep", "role": "train", "line": "v3.1", "decisions": 0, "from": 0, "to": docs[i]["tokens"],
                            "date": "2026-11-23", "note": NOTE}) + "\n"); n += 1
print(f"{n} lines added ({len(ids)} docs)")
