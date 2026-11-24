#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ledger228.py CALIB_META.jsonl TRACE_STATS.json -- append w228's uses to eval/pools/ledger.jsonl (role train, line v3.1):
one line per calib document of onpol-calib-b1 (use qad228-onpol-calib-b1; from 0 to the deepest cut p used as the prompt: the generated
continuation after it is the teacher's, not calib's), and one non-calib line for the trace batch onpol-b1 (use qad228-onpol-b1).
Checks every calib doc is qad / fit / search (pools-v3). Skips (doc, use) pairs already there. CALIB_META = the golden's corpus.meta.jsonl
(only its calib-chat / calib-text lines are read: calib ids and offsets); TRACE_STATS = docs228.py's stats of onpol-b1 (counts only)."""
import json, os, sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
meta, tstats = sys.argv[1:3]; LED = os.path.join(REPO, "eval/pools/ledger.jsonl")
pool = json.load(open(os.path.join(REPO, "eval/pools/pools-v3.json")))["pool"]
have = {(r["doc"], r["use"]) for r in map(json.loads, open(LED))}
to = {}
for r in map(json.loads, open(meta)):
    if r["batch"].startswith("calib-"): to[r["sidx"]] = max(to.get(r["sidx"], 0), r["p"])
ts = next(iter(json.load(open(tstats)).values()))
NOTE = ("#228 (w228, 2026-11-24): on-policy QAD data, golden onpol-calib-b1: the IQ6R-V4 teacher's own continuations (raw completions, "
        "thinking included, every generated token scored) of calib qad / fit / search documents (minus j217, golden-jlong, the jc228 judge): "
        "chat docs cut at assistant turns (prefix <= 8k), text docs continued at a line / paragraph boundary at 30-70 % (prefix <= 6k); "
        "the calib tokens 0..p are context only (unscored); v3/qad228/calib228.py")
n = 0
with open(LED, "a") as f:
    for i in sorted(to):
        assert pool[i] in ("qad", "fit", "search"), (i, pool[i])
        if (i, "qad228-onpol-calib-b1") in have: continue
        f.write(json.dumps({"doc": i, "use": "qad228-onpol-calib-b1", "role": "train", "line": "v3.1", "decisions": 0, "from": 0, "to": to[i],
                            "date": "2026-11-24", "note": NOTE}) + "\n"); n += 1
    if ("noncalib", "qad228-onpol-b1") not in have:
        f.write(json.dumps({"doc": "noncalib", "use": "qad228-onpol-b1", "role": "train", "line": "v3.1", "decisions": 0, "from": None, "to": None,
                            "date": "2026-11-24", "note": f"#228 (w228, 2026-11-24): non-calib corpus onpol-b1 = the IQ6R-V4 teacher's own assistant turns at "
                            f"{ts['docs']} cut points of {ts['sessions']} of our own pi sessions (prefix 8k-128k, the 8k tail kept as context; "
                            f"{ts['gen_rows']} generated rows, thinking included); v3/qad228/cuts228.py, gen228.py, docs228.py. PRIVATE: box2 only"}) + "\n"); n += 1
print(f"{n} lines added ({len(to)} calib docs)")
