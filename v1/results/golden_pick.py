#!/usr/bin/env python3
"""golden_pick.py GOLDEN CORPUS OUTDIR OUTCORPUS DOCIDX... -- a sub-golden of whole documents + matching corpus lines."""
import json, os, sys
import numpy as np
args = sys.argv[1:]
cap = 0
if args[0] == "--maxtok": cap = int(args[1]); args = args[2:]
g, corp, od, oc = args[:4]
caps = {}
idx = None
if args[4:] != ["all"]:
    idx = []
    for x in args[4:]:
        i, _, c = x.partition(":"); idx.append(int(i))
        if c: caps[int(i)] = int(c)
m = json.load(open(f"{g}/kld.json")); top = m["top"]
toks = np.memmap(f"{g}/tokens.i32", np.int32, "r")
tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8)
rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
gen = np.fromfile(f"{g}/gen.u8", np.uint8)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
lines = open(corp).read().splitlines()
if idx is None: idx = list(range(len(m["docs"])))
os.makedirs(od, exist_ok=True)
docs, T, TB, RW, GN, r = [], [], [], [], [], 0
for i in idx:
    d = m["docs"][i]; c = caps.get(i, cap); nt = min(d["tokens"], c) if c else d["tokens"]; n = nt - 1 - d["score_from"]; r0 = d["row0"]
    assert n > 0, (i, nt, d["score_from"])
    T.append(np.asarray(toks[toff[i]:toff[i] + nt])); TB.append(np.asarray(tb[r0:r0 + n]))
    RW.append(np.asarray(rw[r0:r0 + n])); GN.append(gen[r0:r0 + n])
    docs.append({"source": d["source"], "tokens": nt, "score_from": d["score_from"], "row0": r, "orig": i}); r += n
np.concatenate(T).tofile(f"{od}/tokens.i32"); np.concatenate(TB).tofile(f"{od}/top.bin")
np.concatenate(RW).tofile(f"{od}/rows.f32"); np.concatenate(GN).tofile(f"{od}/gen.u8")
mm = dict(m); mm["rows"] = r; mm["docs"] = docs
json.dump(mm, open(f"{od}/kld.json", "w"))
def line(i):
    if i not in caps: return lines[i]
    j = json.loads(lines[i]); j["maxtok"] = caps[i]; return json.dumps(j, ensure_ascii=False)
open(oc, "w").write("\n".join(line(i) for i in idx) + "\n")
