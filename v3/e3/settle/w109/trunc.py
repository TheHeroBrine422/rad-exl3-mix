# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""trunc.py SRC DST L [--level] -- a dev copy of a 27B checkpoint (or an ourq level) cut to its first L decoder layers:
config num_hidden_layers / layer_types cut, index weight_map restricted to layers < L (+ every non-layer tensor whose
shard is present), shards symlinked (no data copied). Timing / identity dev runs of the level passes (#144)."""
import json, os, re, sys

src, dst, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
c = json.load(open(f"{src}/config.json"))
for cc in (c, c.get("text_config") or {}):
    if "num_hidden_layers" in cc:
        cc["num_hidden_layers"] = L
        if "layer_types" in cc: cc["layer_types"] = cc["layer_types"][:L]
json.dump(c, open(f"{dst}/config.json", "w"), indent=1)
ix = json.load(open(f"{src}/model.safetensors.index.json"))
keep = {}
for k, f in ix["weight_map"].items():
    m = re.search(r"\.layers\.(\d+)\.", k)
    if ((m and int(m.group(1)) < L) or not m) and os.path.exists(f"{src}/{f}"): keep[k] = f   # (mtp.layers.* in absent shards: dropped)
ix["weight_map"] = keep
json.dump(ix, open(f"{dst}/model.safetensors.index.json", "w"), indent=1)
for f in set(keep.values()) | {x for x in os.listdir(src) if x in ("stamp.json", "assembled.json")}:
    d = f"{dst}/{f}"
    if not os.path.exists(d): os.symlink(os.path.abspath(f"{src}/{f}"), d)
missing = sorted({f for f in keep.values() if not os.path.exists(f"{src}/{f}")})
print(f"{dst}: {L} layers, {len(keep)} tensors, shards {sorted(set(keep.values()))}" + (f" MISSING {missing}" if missing else ""))
