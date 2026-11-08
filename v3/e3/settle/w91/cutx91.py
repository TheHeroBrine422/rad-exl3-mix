#!/usr/bin/env python3
"""cutx91.py GOLDEN REF_KLD OUTDIR -- golden-search27b's xcorp part re-cut from night1105's xc8k-b1 golden at the piece
indices w84 picked (REF_KLD = the repo's data/w84/golden-search27b-x-kld.json: docs[].orig / xdoc / category). w84's
pickx84.py needs xc8k-b1.meta.jsonl, which went with /data/w84; the pick itself is recorded, so this is pickx84.py's
cut step on the recorded pick. Writes OUTDIR (kld.json, tokens.i32, top.bin, rows.f32, gen.u8) exactly as pickx84.py."""
import json, os, sys
import numpy as np

g, ref, od = sys.argv[1:4]
m = json.load(open(f"{g}/kld.json")); R = json.load(open(ref))
top = m["top"]
toks = np.memmap(f"{g}/tokens.i32", np.int32, "r")
tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8)
rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
os.makedirs(od, exist_ok=True)
docs, T, TB, RW, r = [], [], [], [], 0
for x in R["docs"]:
    i = x["orig"]; d = m["docs"][i]; nt = d["tokens"]; sf = d["score_from"]; k = nt - 1 - sf; assert k > 0
    assert (d["source"], nt, sf) == (x["source"], x["tokens"], x["score_from"]), (x, d)
    T.append(np.asarray(toks[toff[i]:toff[i] + nt])); TB.append(np.asarray(tb[d["row0"]:d["row0"] + k])); RW.append(np.asarray(rw[d["row0"]:d["row0"] + k]))
    docs.append({"source": d["source"], "tokens": nt, "score_from": sf, "row0": r, "orig": i, "xdoc": x["xdoc"], "category": x["category"]}); r += k
np.concatenate(T).tofile(f"{od}/tokens.i32"); np.concatenate(TB).tofile(f"{od}/top.bin"); np.concatenate(RW).tofile(f"{od}/rows.f32")
np.ones(r, np.uint8).tofile(f"{od}/gen.u8")
mm = dict(m); mm["rows"] = r; mm["docs"] = docs
json.dump(mm, open(f"{od}/kld.json", "w"))
open(f"{od}.docs", "w").write(" ".join(str(x["orig"]) for x in R["docs"]) + "\n")
print(f"{od}: {len(docs)} docs, {r} rows", file=sys.stderr)
