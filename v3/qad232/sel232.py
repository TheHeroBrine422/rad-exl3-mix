#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sel232.py XGAP_DIR TOKENIZER_DIR OUT_DIR --exclude META.jsonl [META.jsonl ...] [--seed 232] [--total 5400000] [--maxtok 131072] [--chunk-tokens 2000000]
w232 part A: the xgap training documents of the V3.1 finals (`xg-fin`): xgap-v1 NON-holdout docs, WHOLE (no cut; a document longer than
--maxtok is left out), not in the --exclude metas (matched on `xg_id`: xg-scr1, the gap judge), the target counted in SCORED rows (rows
whose next token the model writes: genmask181.doc_gen, the challenge's rule) per cell share SHARE. Docs are drawn in a seeded order,
round-robin over sources inside a cell, until the cell's target (a doc that would overshoot the target by > 2 % is skipped; 40 skips in a row
end the cell). A cell that runs short (pool exhausted) has its remainder spread pro rata over the cells that still have documents.
Outputs: OUT/xg-fin.jsonl (+ .meta.jsonl: source, category, kind, xg_id, cell, tokens, gen_rows), in chunk order; OUT/xg-fin.c<k>.jsonl
(+ .meta.jsonl): chunks of ~--chunk-tokens tokens, the deepest documents first, the documents > 100k tokens alone in chunk 0 (each chunk is one radiance recording: max_tokens sizes its
max-model-len); OUT/xg-fin.chunks.json; OUT/xg-fin.stats.json (rows / tokens per cell, length histogram)."""
import argparse, collections, json, os, random, sys

import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("xgap"); ap.add_argument("tok"); ap.add_argument("out")
ap.add_argument("--exclude", nargs="+", required=True); ap.add_argument("--seed", type=int, default=232)
ap.add_argument("--total", type=int, default=5_400_000); ap.add_argument("--maxtok", type=int, default=131072)
ap.add_argument("--chunk-tokens", type=int, default=2_000_000); ap.add_argument("--holdout", action="store_true", help="draw the xgap HOLDOUT docs instead (the V3.1 final-read x part)"); ap.add_argument("--mix", help="JSON {gen_rows: {cell: n}} whose cell mix replaces SHARE (golden-jgap's, by scored rows)"); ap.add_argument("--genmask-dir", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()
sys.path.insert(0, a.genmask_dir)
import genmask181 as G

DEEP = 100_000
SHARE = {"long-docs": 30, "multilingual-text": 20, "agent-sessions": 15, "multilingual-chat": 12, "long-chat": 8, "code-langs": 10, "tools-other": 5}
if a.mix: SHARE = {c: v for c, v in json.load(open(a.mix))["gen_rows"].items()}
tk = AutoTokenizer.from_pretrained(a.tok); S, E, A, NL = G.ids(tk)
excl = set()
for p in a.exclude:
    for l in open(p):
        if l.strip(): excl.add(json.loads(l)["xg_id"])
man = [json.loads(l) for l in open(f"{a.xgap}/manifest.jsonl")]
xg = [m for m in man if m["origin"] == "xgap-v1" and bool(m.get("holdout")) == a.holdout and m["id"] not in excl]
print(f"xgap {'holdout' if a.holdout else 'non-holdout'} docs {sum(1 for m in man if m['origin'] == 'xgap-v1' and bool(m.get('holdout')) == a.holdout)}, excluded {len(excl)} ids, candidates {len(xg)}", flush=True)
files = {}
def text(m):
    f = m["file"]
    if f not in files: files[f] = open(f"{a.xgap}/{f}").read().rstrip("\n").split("\n")
    return json.loads(files[f][m["line"]])


class Cell:
    def __init__(self, name):
        self.name = name; rng = random.Random(f"{a.seed}-{name}")
        by = collections.defaultdict(list)
        for m in xg:
            if m["cell"] == name and m["tokens"] <= a.maxtok * 1.1: by[m["source"]].append(m)
        for v in by.values(): rng.shuffle(v)
        self.by = by; self.srcs = sorted(by); rng.shuffle(self.srcs); self.k = 0
        self.rows = 0; self.docs = []; self.skips = 0; self.left_long = 0; self.done = False

    def exhausted(self): return not any(self.by[s] for s in self.srcs)

    def extend(self, target):
        """draw until gen rows >= target (overshoot <= 2 %), the pool is empty, or 40 skips in a row"""
        self.skips = 0
        while self.rows < target and not self.exhausted():
            s = self.srcs[self.k % len(self.srcs)]; self.k += 1
            if not self.by[s]: continue
            m = self.by[s].pop(); j = text(m); p = j["prompt"]
            t = np.asarray(tk(p, add_special_tokens=False)["input_ids"], np.int64); n = len(t)
            if n > a.maxtok: self.left_long += 1; continue
            if n < 64: continue
            w = G.doc_gen(t, S, E, A, NL); g = int(w[1:].sum())
            if g < 16: continue
            if self.rows + g > target * 1.02 and self.rows > 0:
                self.skips += 1
                if self.skips >= 40: break
                continue
            self.skips = 0; self.rows += g
            d = {"source": j.get("source", m["source"]), "category": j.get("category", m["cell"]), "kind": j.get("kind", m.get("kind")), "xg_id": m["id"],
                 "cell": m["cell"], "tokens": n, "gen_rows": g}
            self.docs.append((d, p))


cells = {c: Cell(c) for c in SHARE}
tgt = {c: a.total * SHARE[c] / sum(SHARE.values()) for c in SHARE}
for c in cells:
    cells[c].extend(tgt[c]); print(f"{c}: target {tgt[c]:.0f} gen rows, got {cells[c].rows} in {len(cells[c].docs)} docs (left out as > {a.maxtok}: {cells[c].left_long}, pool left {sum(len(v) for v in cells[c].by.values())})", flush=True)
note = []
for rnd in range(5):                                            # shortfall spread pro rata over the cells that can still grow
    short = {c: tgt[c] - cells[c].rows for c in cells if tgt[c] - cells[c].rows > 0.02 * tgt[c] and cells[c].exhausted()}
    if not short: break
    rem = sum(short.values()); live = [c for c in cells if c not in short and not cells[c].exhausted()]
    if not live: break
    note.append(f"round {rnd}: short {({c: int(v) for c, v in short.items()})}, remainder {int(rem)} spread over {live}")
    ws = sum(SHARE[c] for c in live)
    for c in live: tgt[c] = cells[c].rows + rem * SHARE[c] / ws
    for c in live: cells[c].extend(tgt[c])
    for c in short: tgt[c] = cells[c].rows
for l in note: print(l, flush=True)

docs = sorted((x for c in cells.values() for x in c.docs), key=lambda x: (-x[0]["tokens"], x[0]["xg_id"]))
tot = sum(x[0]["tokens"] for x in docs); chunks = [[]]; acc = 0
for x in docs:
    if chunks[-1] and (acc >= a.chunk_tokens or (x[0]["tokens"] < DEEP < chunks[-1][-1][0]["tokens"])): chunks.append([]); acc = 0   # the docs > DEEP form their own chunk (the TP2 memory risk is confined to it)
    chunks[-1].append(x); acc += x[0]["tokens"]
os.makedirs(a.out, exist_ok=True)
def wr(path, items, with_prompt):
    with open(path + ".part", "w") as f:
        for d, p in items: f.write(json.dumps(dict(d, prompt=p) if with_prompt else d, ensure_ascii=False) + "\n")
    os.replace(path + ".part", path)
info = []; allitems = []
for k, ch in enumerate(chunks):
    fn = f"xg-fin.c{k}.jsonl"; items = []
    for d, p in ch: items.append((dict(d, chunk=k), p))
    wr(f"{a.out}/{fn}", items, True); wr(f"{a.out}/{fn}.meta.jsonl", items, False)
    info.append({"k": k, "file": fn, "docs": len(ch), "tokens": sum(d["tokens"] for d, _ in ch), "max_tokens": max(d["tokens"] for d, _ in ch),
                 "rows": sum(d["tokens"] - 1 for d, _ in ch), "gen_rows": sum(d["gen_rows"] for d, _ in ch)})
    allitems += items
wr(f"{a.out}/xg-fin.jsonl", allitems, True); wr(f"{a.out}/xg-fin.jsonl.meta.jsonl", allitems, False)
json.dump(info, open(f"{a.out}/xg-fin.chunks.json", "w"), indent=1)
edges = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131073]
st = {"tokens": tot, "gen_rows": sum(x[0]["gen_rows"] for x in docs), "docs": len(docs), "notes": note, "cells": {}, "length_hist": {}}
for c in cells:
    dd = [x[0] for x in cells[c].docs]
    st["cells"][c] = {"docs": len(dd), "tokens": sum(d["tokens"] for d in dd), "gen_rows": sum(d["gen_rows"] for d in dd), "target": int(a.total * SHARE[c] / sum(SHARE.values())),
                      "share_pct": 0.0, "left_out_over_maxtok": cells[c].left_long, "pool_left": sum(len(v) for v in cells[c].by.values())}
for c in st["cells"]: st["cells"][c]["share_pct"] = round(100 * st["cells"][c]["gen_rows"] / st["gen_rows"], 2)
for lo, hi in zip(edges, edges[1:]):
    sel = [x[0] for x in docs if lo <= x[0]["tokens"] < hi]
    st["length_hist"][f"{lo}-{hi - 1}"] = {"docs": len(sel), "tokens": sum(d["tokens"] for d in sel), "gen_rows": sum(d["gen_rows"] for d in sel)}
json.dump(st, open(f"{a.out}/xg-fin.stats.json", "w"), indent=1)
print(json.dumps(st, indent=1)); print("chunks:", [(c["k"], c["docs"], c["tokens"], c["max_tokens"]) for c in info])
