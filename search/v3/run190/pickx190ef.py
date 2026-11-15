# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pickx190cd.py PLAN.json NAME XVIEW OUTDIR OUTCORPUS CALIB_KLD -- the x part NAME of plan190cd.json (#190-02, w176). Its pieces come
from several xcorp batches (xc8k-b1 and xc8k-b2-c0..c2): each piece has its batch, its index in that batch, and its cap. The output
is pickx186.py's x part, extended to several batches: tokens.i32, top.bin and rows.f32 cut to the pieces (a prefix cut keeps the rows
exact: the model is causal), gen.u8 all ones (genmask181.py makes the board mask afterwards, build190cd.py), kld.json with
orig = the piece's index in its batch, xdoc (the corpus doc id, the global identity), category and batch. OUTCORPUS: one line per
piece, "{}" or {"maxtok": n} for a cut piece. Pieces are ordered by (batch, idx). The header (format, corpus, top, n_vocab) is
xc8k-b1's and model is the calib golden's, as build190's build_x. XVIEW: the xview dir (per batch: kld.json, tokens.i32, top.bin,
rows.f32, corpus.meta.jsonl)."""
import json, os, sys

import numpy as np

PLAN, NAME, XV, OD, OC, CALKLD = sys.argv[1:7]
ORDER = {"xc8k-b1": 0, "xc8k-b2-c0": 1, "xc8k-b2-c1": 2, "xc8k-b2-c2": 3}
plan = json.load(open(PLAN)); E = [p for p in plan["x"] if p["name"] == NAME]
assert len(E) == 1, f"no x part {NAME} in {PLAN}"
pieces = sorted(E[0]["pieces"], key=lambda p: (ORDER[p["batch"]], p["idx"]))
model = json.load(open(CALKLD))["model"]
cache = {}


def batch(b):
    if b not in cache:
        d = f"{XV}/{b}"
        m = json.load(open(f"{d}/kld.json")); M = [json.loads(l) for l in open(f"{d}/corpus.meta.jsonl")]
        assert len(M) == len(m["docs"]), (b, len(M), len(m["docs"]))
        toks = np.memmap(f"{d}/tokens.i32", np.int32, "r"); top = m["top"]
        tb = np.memmap(f"{d}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8)
        rw = np.memmap(f"{d}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
        toff = np.concatenate([[0], np.cumsum([x["tokens"] for x in m["docs"]])])
        cache[b] = (m, M, toks, tb, rw, toff)
    return cache[b]


os.makedirs(OD, exist_ok=True)
docs, T, TB, RW, lines, r, ncut = [], [], [], [], [], 0, 0
head = None
for p in pieces:
    m, M, toks, tb, rw, toff = batch(p["batch"]); i = p["idx"]
    if head is None and p["batch"] == "xc8k-b1": head = m
    d = m["docs"][i]
    assert M[i]["doc"] == p["doc"], (p, M[i]["doc"])                    # the xsupply row is this piece
    assert M[i]["tokens"] == d["tokens"] and M[i]["source"] == d["source"], (p, "meta differs from the kld")
    nt = d["tokens"]; sf = d["score_from"]; k = nt - 1 - sf; assert k > 0
    c = int(p.get("cap") or 0); line = "{}"
    if c and k > c:
        k = c; nt = sf + 1 + c; ncut += 1; line = json.dumps({"maxtok": nt})
    T.append(np.asarray(toks[toff[i]:toff[i] + nt])); TB.append(np.asarray(tb[d["row0"]:d["row0"] + k]))
    RW.append(np.asarray(rw[d["row0"]:d["row0"] + k]))
    docs.append({"source": d["source"], "tokens": nt, "score_from": sf, "row0": r, "orig": i, "xdoc": M[i]["doc"],
                 "category": M[i]["category"], "batch": p["batch"]})
    lines.append(line); r += k
head = head or next(iter(cache.values()))[0]
np.concatenate(T).astype(np.int32).tofile(f"{OD}/tokens.i32"); np.concatenate(TB).tofile(f"{OD}/top.bin")
np.concatenate(RW).tofile(f"{OD}/rows.f32"); np.ones(r, np.uint8).tofile(f"{OD}/gen.u8")
mm = {k: head[k] for k in head if k not in ("docs", "rows")}
mm.update({"model": model, "rows": r, "docs": docs})
json.dump(mm, open(f"{OD}/kld.json", "w"))
open(OC, "w").write("\n".join(lines) + "\n")
open(f"{OD}.docs", "w").write(" ".join(f"{p['batch']}:{p['idx']}" for p in pieces) + "\n")
print(f"{OD}: {len(pieces)} pieces from {len(set(p['batch'] for p in pieces))} batches, {r} rows ({ncut} cut to their cap)", file=sys.stderr)
