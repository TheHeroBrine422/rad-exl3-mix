#!/usr/bin/env python3
"""subset.py META OUT CAT[,CAT...] -- the documents of a golden-style META (kld.json + tokens.i32) whose category is in
the list, as a new META (rows / row0 renumbered). d1111 amendment 3's category arms."""
import json, os, sys
import numpy as np
src, out, cats = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
m = json.load(open(f"{src}/kld.json")); t = np.fromfile(f"{src}/tokens.i32", np.int32)
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
os.makedirs(out, exist_ok=True); docs, T, row0 = [], [], 0
for i, d in enumerate(m["docs"]):
    if d.get("category") not in cats: continue
    T.append(t[off[i]:off[i + 1]]); docs.append(dict(d, row0=row0)); row0 += d["tokens"] - 1 - d["score_from"]
np.concatenate(T).tofile(f"{out}/tokens.i32")
if os.path.exists(f"{src}/gen.u8"):
    g = np.fromfile(f"{src}/gen.u8", np.uint8); s = []
    for i, d in enumerate(m["docs"]):
        if d.get("category") in cats: s.append(g[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]])
    np.concatenate(s).tofile(f"{out}/gen.u8")
with open(f"{out}/kld.json.part", "w") as fh: json.dump(dict(m, docs=docs, rows=row0), fh)   # .part + rename: kld.json = done
os.replace(f"{out}/kld.json.part", f"{out}/kld.json")
print("SUBSET", len(docs), "docs", sum(d["tokens"] for d in docs), "tokens")
