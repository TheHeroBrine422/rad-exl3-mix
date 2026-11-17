#!/usr/bin/env python3
# progcheck.py GOLDEN PROGRESS RUN_BASE -- checks a --kld-progress run (w201): every document's line (kl_sum, rows,
# kl_off into PROGRESS.kl) against the rows file of a run without progress (RUN_BASE.json.rows, same golden).
# Run with -I. Prints one JSON object.
import json
import sys

import numpy as np

gold, prog, base = sys.argv[1:4]
m = json.load(open(f"{gold}/kld.json"))
K = np.fromfile(base + ".json.rows", np.float32).reshape(-1, 4)
kl_all = np.fromfile(prog + ".kl", np.float32)
lines = [json.loads(l) for l in open(prog) if l.endswith("\n")]
seen = sorted(j["doc"] for j in lines)
worst_sum = 0.0
worst_kl = 0.0
bad = []
for j in lines:
    d = m["docs"][j["doc"]]
    row0, n = d["row0"], j["rows"]
    if n != d["tokens"] - 1 - d["score_from"]:
        bad.append({"doc": j["doc"], "why": "rows"})
        continue
    ref = K[row0:row0 + n, 0].astype(np.float64)
    worst_sum = max(worst_sum, abs(j["kl_sum"] - ref.sum()))
    seg = kl_all[j["kl_off"] // 4: j["kl_off"] // 4 + n]
    worst_kl = max(worst_kl, float(np.max(np.abs(seg.astype(np.float64) - ref))) if n else 0.0)
print(json.dumps({
    "progress_lines": len(lines),
    "distinct_docs": len(set(seen)),
    "docs_in_golden": len(m["docs"]),
    "order_docs_first_5": [j["doc"] for j in lines[:5]],
    "max_abs_kl_sum_diff_vs_rows": worst_sum,
    "max_abs_perpos_kl_diff_vs_rows": worst_kl,
    "bad": bad[:5],
}, indent=1))
