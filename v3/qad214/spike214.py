#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""spike214.py RUN_DIR STEP [STEP ...] -- the batches of a #214 run's plan steps (train.jsonl "step" numbers): rebuilds the plan from the run's own arguments
(run.log's first line) with qad200.build_data and prints, per stream, each segment: corpus, document (calib: the calib-corpus index = the ledger's doc id,
category, source), token window, gen rows. For looking at a training-KL spike."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad200"))
import qad200 as Q  # noqa: E402

run, steps = sys.argv[1], [int(x) for x in sys.argv[2:]]
first = open(os.path.join(run, "run.log")).readline()
args = json.loads(first[first.index("{"):first.rindex("}") + 1])
a = Q.parser().parse_args(["out", "--model", "x", "--level", "x", "--chunks", "x"]); a.__dict__.update(args); a.out = "/tmp/spike214-" + os.path.basename(run.rstrip("/")); os.makedirs(a.out, exist_ok=True)
P = Q.build_data(a, Q.Dist(), lambda s: None)
C = P["C"]
cal = [json.loads(l) for l in open("/data/q27/challenge/corpus/calib.jsonl").read().rstrip("\n").split("\n")]
for s in steps:
    print(f"== plan step {s} (0-based {s - 1}), rows so far {int(P['cum'][s - 1])}")
    tot = {}
    for si, segs in enumerate(P["steps"][s - 1]):
        for (it, sa, sb) in segs:
            gi, doc, n, nr = C.items[it]; g = C.g[gi]; tag = C.tags[gi]; rows = P["rows_of"][(it, sa, sb)]
            tot[tag] = tot.get(tag, 0) + rows
            extra = ""
            if g.name == "calib": extra = f" calib doc {doc} category {cal[doc]['category']} source {cal[doc].get('source', g.docs[doc].get('source'))}"
            else: extra = f" doc {doc} source {g.docs[doc].get('source')}"
            print(f"  stream {si:2d} {g.name:12s}{extra}  tokens [{sa}, {sb}) of {n}  gen rows {rows}")
    print("  rows by corpus:", tot)
