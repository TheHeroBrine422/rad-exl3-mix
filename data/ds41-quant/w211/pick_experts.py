#!/usr/bin/env python3
"""pick_experts.py COUNTS.json CALIB_DIR OUT.json -- the w211 expert set. Per layer (5, 20, 35), experts of shard s0 (0-191) ranked by
routed-token count (counts.json, 100M tokens): hot = ranks 0, 1 (top 5 % = ranks 0-9); medium = ranks 95, 96 (median);
cold = the two lowest-count experts among the bottom 5 % (ranks 182-191) that have >= 1000 kept rows (stats n)."""
import json, sys
from safetensors import safe_open
counts = json.load(open(sys.argv[1]))["counts"]; calib = sys.argv[2]
out = {}
for L in (5, 20, 35):
    c = counts[L][:192]; order = sorted(range(192), key=lambda e: (-c[e], e))
    f = safe_open(f"{calib}/experts/L{L:02d}-s0.safetensors", "np")
    kept = {e: int(f.get_tensor(f"e{e:03d}.stats")[0]) for e in order}
    cold = [e for e in reversed(order[-10:]) if kept[e] >= 1000][:2]
    sel = {"hot": order[:2], "medium": [order[95], order[96]], "cold": cold}
    out[str(L)] = {k: [{"e": e, "rank": order.index(e), "count": c[e], "kept_rows": kept[e]} for e in v] for k, v in sel.items()}
    out[str(L)]["bottom10_kept_rows"] = {str(e): kept[e] for e in order[-10:]}
json.dump(out, open(sys.argv[3], "w"), indent=1)
for L, d in out.items():
    print(L, {k: [(x["e"], x["count"], x["kept_rows"]) for x in v] for k, v in d.items() if k != "bottom10_kept_rows"}, d["bottom10_kept_rows"])
