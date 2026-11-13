#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""a8docs.py GOLDEN OUTDIR NAME=ROWS... -- (w135, #174, ds8) per-document KL sums over the golden's gen positions
(rxrun.hostd.doc_sums, as hostd and w132's docsums132.py) of radiance --kld-out rows, written as the coordinator's docs
files (OUTDIR/<NAME>-<slice>.json, {"m": [[KL sum, positions], ...]}) so calobs.py pairs them like search results."""
import json, os, sys
sys.path.insert(0, "/data/q27/stage/code/pipeline")
from rxrun.hostd import doc_sums   # noqa: E402

g, out = sys.argv[1], sys.argv[2]; sl = os.path.basename(g.rstrip("/")).replace("golden-", "")
os.makedirs(out, exist_ok=True)
for kv in sys.argv[3:]:
    n, rows = kv.split("=", 1)
    d, k = doc_sums(g, rows)
    json.dump({"m": d, "kld": float(k.mean()), "positions": int(len(k))}, open(f"{out}/{n}-{sl}.json", "w"))
    print(n, len(d), sum(1 for x in d if x[1] == 0), round(float(k.mean()), 6))
