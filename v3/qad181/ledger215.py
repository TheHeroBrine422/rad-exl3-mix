#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ledger215.py CALIB POOLS OUTDIR -- w215's calib document uses as kld.json-style files for eval/pools/ledger_add.py:
OUTDIR/qad215-train.json (the documents eq215's 20 steps train on: V 6 x Q 1024, seed 0, the qad pool; per document
positions 0 .. the last one trained) and OUTDIR/qad215-check.json (eval215's held-out sanity docs: the last 40 of the
seeded order, first 2,048 tokens)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import data181 as D  # noqa: E402

calib, pools, out = sys.argv[1:4]
C = D.Corpus([], 8192, calib=calib, pools=pools, pool="qad")
order = np.random.default_rng(0).permutation(len(C.items))
steps = D.plan(C.items, order, 6, 1024, max_steps=20)
hi = {}
for st in steps:
    for segs in st:
        for (it, a, b) in segs: i = C.items[it][1]; hi[i] = max(hi.get(i, 0), b)
m = json.load(open(os.path.join(calib, "kld.json")))
def doc(i, b): return {"orig": int(i), "score_from": 0, "tokens": int(b) + 1}
for name, docs in (("qad215-train", [doc(i, b) for i, b in sorted(hi.items())]),
                   ("qad215-check", [doc(C.items[k][1], min(2048, C.items[k][2])) for k in sorted(order[-40:], key=lambda k: C.items[k][1])])):
    json.dump({"corpus_fnv1a64": m["corpus_fnv1a64"], "created": "2026-11-21", "docs": docs}, open(os.path.join(out, name + ".json"), "w"))
    print(name, len(docs), "docs")
