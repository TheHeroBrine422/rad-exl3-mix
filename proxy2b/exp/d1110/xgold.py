#!/usr/bin/env python3
"""xgold.py XMETA OUT N SEED [CAP] [EXCLUDE_GOLDEN_DIR...] -- d1110: a golden-format document set (kld.json / tokens.i32 / gen.u8, every
position scored) from N xcorp documents, stratified by category in proportion to xcorp's mix, each cut to CAP tokens
(default 2048). mkgold.py OUT OUT2 0..N-1 then adds the bf16 reference. Out of calib's documents entirely."""
import json, os, random, sys
import numpy as np
src, out, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); cap = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
EXC = {d['orig'] for g in sys.argv[6:] if os.path.exists(f'{g}/kld.json') for d in json.load(open(f'{g}/kld.json'))['docs']}
m = json.load(open(f"{src}/kld.json")); toks = np.memmap(f"{src}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
by = {}
for i, d in enumerate(m["docs"]):
    if d["tokens"] >= 64 and i not in EXC: by.setdefault(d.get("category", "?"), []).append(i)
tot = sum(len(v) for v in by.values()); rng = random.Random(seed); pick = []
for c, v in sorted(by.items()):
    rng.shuffle(v); pick += v[:max(1, round(n * len(v) / tot))]
pick = sorted(pick[:n]) if len(pick) >= n else sorted(pick)
os.makedirs(out, exist_ok=True); docs, T, row0 = [], [], 0
for i in pick:
    t = np.asarray(toks[off[i]:off[i] + min(cap, m["docs"][i]["tokens"])], np.int32); T.append(t)
    docs.append({"source": m["docs"][i].get("source"), "category": m["docs"][i].get("category"), "tokens": int(len(t)),
                 "score_from": 0, "row0": row0, "orig": int(i)}); row0 += len(t) - 1
np.concatenate(T).tofile(f"{out}/tokens.i32"); np.ones(row0, np.uint8).tofile(f"{out}/gen.u8")
with open(f"{out}/kld.json.part", "w") as fh: json.dump({"format": "d1110-xcorp", "rows": row0, "docs": docs}, fh)   # .part + rename: kld.json = done
os.replace(f"{out}/kld.json.part", f"{out}/kld.json")
print("XGOLD DONE", len(docs), "docs", row0, "rows")
