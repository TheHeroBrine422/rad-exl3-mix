#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""docsums158.py ROOT SLICE GOLDEN [OUT.json] -- (w158, #173-03, ds8) per-document KL sums (rxrun.hostd.doc_sums) of every
ROOT/scores/<tag>-<SLICE>.json.rows -> {tag: {"docs": [[sum, n], ..], "kld", "positions"}} (JSON)."""
import glob, json, os, sys
sys.path.insert(0, "/data/q27/stage/code/pipeline")
from rxrun.hostd import doc_sums  # noqa: E402
root, sl, g = sys.argv[1:4]; out = {}
for r in sorted(glob.glob(f"{root}/scores/*-{sl}.json.rows")):
    t = os.path.basename(r)[:-len(f"-{sl}.json.rows")]
    if not os.path.exists(r[:-5]): continue
    d, kl = doc_sums(g, r); out[t] = {"docs": d, "kld": float(kl.mean()), "positions": int(len(kl))}
json.dump(out, open(sys.argv[4], "w") if len(sys.argv) > 4 else sys.stdout)
