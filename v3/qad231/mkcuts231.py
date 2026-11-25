#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkcuts231.py SRC_GOLDEN SRC_GENMASK OUT_DIR [--sessions 6] [--cuts 3] [--min-tokens 20000] [--Q 1024] [--seed 0] -- w231: a multi-cut test golden for the
branch-vs-separate GPU check, cut from an existing golden's long documents (no new recording: a cut = the document's first P + g tokens, scored from P,
with the source's own rows for those positions, which are the teacher's distributions after the true prefix). Each session = one source document,
`--cuts` cuts at random depths; every cut's length P + g is a multiple of Q (so the separate documents and the branch pieces fall on the same segment
boundaries). Writes OUT_DIR as a golden (kld.json, tokens.i32, top.bin, rows.f32, gen.u8 = the source's gen mask for those rows) + corpus.meta.jsonl
({"session", "cut"}). The source's documents may be private traces: OUT_DIR stays on our machines."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D                                      # noqa: E402

a = sys.argv[1:]; src, gm, out = a[:3]
opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
NS, NC, MT, Q, seed = opt("--sessions", 6), opt("--cuts", 3), opt("--min-tokens", 20000), opt("--Q", 1024), opt("--seed", 0)
g = D.Gold(src, genfile=gm); rg = np.random.default_rng(seed)
cand = [i for i, d in enumerate(g.docs) if d["tokens"] >= MT and d["score_from"] == 0]
assert len(cand) >= NS, f"{len(cand)} docs >= {MT} tokens"
pick = sorted(rg.choice(cand, NS, replace=False))
os.makedirs(out, exist_ok=True)
toks, tops, rows, gens, docs, meta, r0 = [], [], [], [], [], [], 0
for s, i in enumerate(pick):
    n_i = g.ntok(i)
    for k, P in enumerate(sorted(rg.choice(np.arange(2 * Q, n_i - 2 * Q), NC, replace=False))):
        n = min(n_i - 1, (int(P) + int(rg.integers(300, 1500))) // Q * Q + Q) // Q * Q; P = int(min(P, n - 64))
        r = g.docs[i]["row0"] + np.arange(P, n - 1)              # rows of positions P .. n - 2 (score_from 0 in the source)
        toks.append(g.ids(i, 0, n)); tops.append(np.asarray(g.top[r[0]:r[-1] + 1])); rows.append(np.asarray(g.rows[r[0]:r[-1] + 1]))
        gens.append(np.asarray(g.gen[r[0]:r[-1] + 1]) if g.gen is not None else np.ones(len(r), np.uint8))
        docs.append({"tokens": int(n), "score_from": P, "row0": r0}); r0 += len(r); meta.append({"session": f"s{s}", "cut": k, "src_doc": int(i)})
np.concatenate(toks).astype(np.int32).tofile(f"{out}/tokens.i32"); np.concatenate(tops).tofile(f"{out}/top.bin")
np.concatenate(rows).astype(np.float32).tofile(f"{out}/rows.f32"); np.concatenate(gens).astype(np.uint8).tofile(f"{out}/gen.u8")
m = dict(g.m); m.update(docs=docs, rows=r0, corpus=f"mkcuts231 of {os.path.basename(src.rstrip('/'))}")
json.dump(m, open(f"{out}/kld.json", "w"))
with open(f"{out}/corpus.meta.jsonl", "w") as f: f.writelines(json.dumps(x) + "\n" for x in meta)
print(f"{len(docs)} cuts of {NS} sessions, {r0} rows, {sum(d['tokens'] for d in docs)} tokens -> {out}")
