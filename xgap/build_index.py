#!/usr/bin/env python3
"""build_index.py -- the dedup index: normalized-content hashes + long-line hashes of calib, public and xcorp-v1.
Public is read for hashes only (nothing of it is copied or stored but the hashes). Every `pisess*` file of xcorp is
refused (the user's private sessions never enter this corpus)."""
import glob, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

docs, lines = {}, {}


def add(origin, prompt):
    b = ORIGIN_BIT[origin]
    h = norm_hash(prompt); docs[h] = docs.get(h, 0) | b
    seen = set()
    for l in long_lines(prompt):
        k = lhash(l)
        if k in seen: continue
        seen.add(k)
        v = lines.get(k, 0)
        lines[k] = ((v & 7) | b) | (min((v >> 3) + 1, 1 << 20) << 3)


t0 = time.time(); n = {}
for origin, files in (("calib", [CALIB]), ("public", [PUBLIC]), ("xcorp", sorted(glob.glob(f"{XCORP}/*.jsonl")))):
    for f in files:
        assert "pisess" not in os.path.basename(f), f
        for l in open(f):
            add(origin, json.loads(l)["prompt"]); n[origin] = n.get(origin, 0) + 1
    print(origin, n[origin], f"{time.time() - t0:.0f}s", len(lines), flush=True)
tmp = INDEX_PKL + ".part"
with open(tmp, "wb") as f: pickle.dump({"docs": docs, "lines": lines, "ndocs": n}, f)
os.replace(tmp, INDEX_PKL)
print("DONE", n, "docs", len(docs), "lines", len(lines))
