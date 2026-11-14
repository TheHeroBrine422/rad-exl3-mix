# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pickx186.py GOLDEN CORPUS_STUB META PLAN.json NAME OUTDIR OUTCORPUS -- the x part of a #186 slice from an explicit piece
list (draw186.py's plan["x"][NAME]: "idx" and "caps"). The output is pickx.py's (eval/slices/pickx.py): kld.json docs with
orig = the piece's index in GOLDEN, xdoc, category; tokens.i32, top.bin, rows.f32 cut to the pieces; gen.u8 all ones
(genmask181.py makes the board mask afterwards); OUTDIR.docs; OUTCORPUS one line per piece (the stub, with "maxtok" for
the cut pieces). Prefix cuts keep the golden's rows exact (causal model), as pickx does."""
import json, os, sys

import numpy as np

g, corp, meta, plan_path, name, od, oc = sys.argv[1:8]
plan = json.load(open(plan_path)); E = [p for p in plan["x"] if p["name"] == name]
assert len(E) == 1, f"no x part {name} in {plan_path}"
pick = sorted(E[0]["idx"]); caps = {int(k): v for k, v in E[0]["caps"].items()}
M = [json.loads(l) for l in open(meta)]
m = json.load(open(f"{g}/kld.json")); assert len(M) == len(m["docs"]), (len(M), len(m["docs"]))
for x, d in zip(M, m["docs"]): assert x["tokens"] == d["tokens"] and x["source"] == d["source"], (x, d)
top = m["top"]
toks = np.memmap(f"{g}/tokens.i32", np.int32, "r")
tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8)
rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
lines = open(corp).read().rstrip("\n").split("\n"); assert len(lines) == len(M)
os.makedirs(od, exist_ok=True)
docs, T, TB, RW, out_lines, r, ncut = [], [], [], [], [], 0, 0
for i in pick:
    d = m["docs"][i]; nt = d["tokens"]; sf = d["score_from"]; k = nt - 1 - sf; assert k > 0
    c = caps.get(i, 0)
    line = lines[i]
    if c and k > c:
        k = c; nt = sf + 1 + c; ncut += 1
        j = json.loads(line); j["maxtok"] = nt; line = json.dumps(j, ensure_ascii=False)
    T.append(np.asarray(toks[toff[i]:toff[i] + nt])); TB.append(np.asarray(tb[d["row0"]:d["row0"] + k])); RW.append(np.asarray(rw[d["row0"]:d["row0"] + k]))
    docs.append({"source": d["source"], "tokens": nt, "score_from": sf, "row0": r, "orig": i, "xdoc": M[i]["doc"], "category": M[i]["category"]})
    out_lines.append(line); r += k
np.concatenate(T).tofile(f"{od}/tokens.i32"); np.concatenate(TB).tofile(f"{od}/top.bin"); np.concatenate(RW).tofile(f"{od}/rows.f32")
np.ones(r, np.uint8).tofile(f"{od}/gen.u8")
mm = dict(m); mm["rows"] = r; mm["docs"] = docs
json.dump(mm, open(f"{od}/kld.json", "w"))
open(oc, "w").write("\n".join(out_lines) + "\n")
open(f"{od}.docs", "w").write(" ".join(map(str, pick)) + "\n")
print(f"{od}: {len(pick)} pieces, {r} rows ({ncut} cut to their category cap)", file=sys.stderr)
