#!/usr/bin/env python3
"""w84q.py ORDER RXQ-ARGS... -- `python -m rxq quant RXQ-ARGS` with blkorder84 (route B, ../w84.md) installed on the
quantize module rxq sets up. ORDER = order82.json or "ident". Needs PYTHONPATH=<code>/pipeline (as rxq).
Writes OUT/blkorder84.json (order, the count of reordered tensors) next to rxq's t-<key>.safetensors."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blkorder84
from rxq import engine

order = sys.argv[1]
_setup = engine.setup


def setup(*a, **kw):
    Q = _setup(*a, **kw)
    blkorder84.install(Q, order)
    return Q


engine.setup = setup
from rxq.__main__ import main

argv = ["quant"] + sys.argv[2:]
out = argv[argv.index("-o") + 1]
main(argv)
prev = json.load(open(f"{out}/blkorder84.json")) if os.path.exists(f"{out}/blkorder84.json") else []
prev.append({"order": order, "stored_order": None if order == "ident" else json.load(open(order))["stored_order"],
             "reordered_this_run": blkorder84.count()})
json.dump(prev, open(f"{out}/blkorder84.json", "w"), indent=1)
print(f"w84q: {blkorder84.count()} tensors reordered this run", flush=True)
