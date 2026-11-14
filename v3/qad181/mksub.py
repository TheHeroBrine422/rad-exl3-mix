#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mksub.py GOLDEN OUT DOC:CAP ... [--all-rows] -- a sub-golden of the first CAP tokens of documents of any radiance-kld-top golden
(top-256 / top-512, with or without gen.u8): kld.json, tokens.i32, top.bin, rows.f32, gen.u8 (the source's gen rows,
ones when it has none; --all-rows: ones, the source's gen kept in gen_src.u8). Prefix cuts keep the rows exact (causal model). For student-vs-radiance checks (#181)."""
import json, os, sys

import numpy as np

allrows = "--all-rows" in sys.argv
if allrows: sys.argv.remove("--all-rows")
g, od = sys.argv[1], sys.argv[2]
m = json.load(open(f"{g}/kld.json")); K = m.get("top", 512); R = m["rows"]
tok = np.memmap(f"{g}/tokens.i32", np.int32, "r"); tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(R, K * 8)
rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(R, 3)
gen = np.fromfile(f"{g}/gen.u8", np.uint8) if os.path.exists(f"{g}/gen.u8") and not allrows else np.ones(R, np.uint8)
gen_src = np.fromfile(f"{g}/gen.u8", np.uint8) if os.path.exists(f"{g}/gen.u8") else np.ones(R, np.uint8)
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
os.makedirs(od, exist_ok=True)
T, TB, RW, GN, GS, docs = [], [], [], [], [], []; r0 = 0
for x in sys.argv[3:]:
    i, c = map(int, x.split(":")); d = m["docs"][i]; n = min(c, d["tokens"])
    T.append(np.asarray(tok[off[i]:off[i] + n]))
    a = d["row0"]; nr = max(0, n - 1 - d["score_from"])
    TB.append(np.asarray(tb[a:a + nr])); RW.append(np.asarray(rw[a:a + nr])); GN.append(gen[a:a + nr]); GS.append(gen_src[a:a + nr])
    docs.append({**{k: v for k, v in d.items() if k not in ("tokens", "row0")}, "tokens": n, "row0": r0, "orig": i}); r0 += nr
np.concatenate(T).astype(np.int32).tofile(f"{od}/tokens.i32"); np.concatenate(TB).tofile(f"{od}/top.bin")
np.concatenate(RW).astype(np.float32).tofile(f"{od}/rows.f32"); np.concatenate(GN).astype(np.uint8).tofile(f"{od}/gen.u8"); np.concatenate(GS).astype(np.uint8).tofile(f"{od}/gen_src.u8")
json.dump({**{k: v for k, v in m.items() if k != "docs"}, "rows": r0, "docs": docs, "sub_of": os.path.abspath(g)}, open(f"{od}/kld.json", "w"))
print(f"{od}: {len(docs)} docs, {r0} rows")
