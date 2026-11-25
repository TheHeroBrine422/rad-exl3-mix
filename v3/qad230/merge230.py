#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""merge230.py NAME DOCSDIR RECDIR FINAL [--corpus-out PATH] [--mask-out PATH] -- w230 step 5: merge the chunk goldens
RECDIR/NAME-c<k> (radiance --kld-record of DOCSDIR/NAME.c<k>.jsonl, listed in DOCSDIR/NAME.chunks.json) into ONE golden FINAL
whose corpus is DOCSDIR/NAME.jsonl (the chunks concatenated in chunk order). tokens.i32 / rows.f32 / top.bin are concatenated, kld.json
is the first chunk's with docs / rows / corpus / corpus_fnv1a64 rewritten (doc `orig` = the index in the merged corpus, row0 offset),
gen.u8 = all ones (every row a generated token), corpus.meta.jsonl = the merged meta. Checks: each chunk's docs / tokens /
score_from / rows against its meta before merging. With --corpus-out the merged corpus (+ .meta.jsonl) is also copied there and kld.json's `corpus` names it (slices/golden-jt*.jsonl
layout); --mask-out writes the gen mask (the --plan-ext layout). FINAL is written as FINAL.part and renamed."""
import argparse, json, os, shutil, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("docs"); ap.add_argument("rec"); ap.add_argument("final")
ap.add_argument("--corpus-out"); ap.add_argument("--mask-out")
a = ap.parse_args()


def fnv1a64(path):
    """radiance's corpus hash (core/kld.cpp): FNV-1a 64 chained over every getline() line without its newline, offset basis 1469598103934665603"""
    h = 1469598103934665603
    lines = open(path, "rb").read().split(b"\n")
    if lines and lines[-1] == b"": lines.pop()
    for l in lines:
        for c in l: h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


chunks = json.load(open(f"{a.docs}/{a.name}.chunks.json")); corpus = f"{a.docs}/{a.name}.jsonl"
tmp = a.final + ".part"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
docs, rows, base = [], 0, None
meta_all = [json.loads(l) for l in open(corpus + ".meta.jsonl")]
with open(f"{tmp}/tokens.i32", "wb") as ft, open(f"{tmp}/rows.f32", "wb") as fr, open(f"{tmp}/top.bin", "wb") as fp:
    for ch in chunks:
        g = f"{a.rec}/{a.name}-c{ch['k']}"; assert os.path.exists(f"{g}/.done"), f"chunk {g} not recorded"
        m = json.load(open(f"{g}/kld.json")); base = base or m
        cm = [json.loads(l) for l in open(f"{a.docs}/{a.name}.c{ch['k']}.jsonl.meta.jsonl")]
        assert len(m["docs"]) == len(cm) == ch["docs"], (g, len(m["docs"]), len(cm), ch["docs"])
        for d, x in zip(m["docs"], cm): assert d["tokens"] == x["tokens"] and d["score_from"] == x["score_from"], (g, d, x)
        assert m["rows"] == sum(x["gen_rows"] for x in cm) == ch["rows"], (g, m["rows"], ch["rows"])
        assert m["top"] == base["top"] and m["n_vocab"] == base["n_vocab"]
        for d in m["docs"]:
            docs.append({"source": d["source"], "tokens": d["tokens"], "score_from": d["score_from"], "row0": rows + d["row0"], "orig": len(docs)})
        for fn, out in (("tokens.i32", ft), ("rows.f32", fr), ("top.bin", fp)):
            with open(f"{g}/{fn}", "rb") as f: shutil.copyfileobj(f, out, 1 << 24)
        rows += m["rows"]
assert len(docs) == len(meta_all), (len(docs), len(meta_all))
for i, (d, x) in enumerate(zip(docs, meta_all)): assert x["doc"] == i and d["tokens"] == x["tokens"], (i, d, x)
np.ones(rows, np.uint8).tofile(f"{tmp}/gen.u8")
if a.corpus_out:
    shutil.copy(corpus, a.corpus_out + ".part"); os.replace(a.corpus_out + ".part", a.corpus_out)
    shutil.copy(corpus + ".meta.jsonl", a.corpus_out + ".meta.jsonl.part"); os.replace(a.corpus_out + ".meta.jsonl.part", a.corpus_out + ".meta.jsonl")
k = dict(base); k["docs"] = docs; k["rows"] = rows; k["corpus"] = a.corpus_out or corpus; k["corpus_fnv1a64"] = f"{fnv1a64(corpus):016x}"
json.dump(k, open(f"{tmp}/kld.json", "w"))
shutil.copy(corpus + ".meta.jsonl", f"{tmp}/corpus.meta.jsonl"); open(f"{tmp}/.done", "w").close()
shutil.rmtree(a.final, ignore_errors=True); os.replace(tmp, a.final)
if a.mask_out: shutil.copy(f"{a.final}/gen.u8", a.mask_out + ".part"); os.replace(a.mask_out + ".part", a.mask_out)
print(f"merged {a.name}: {len(docs)} docs, {rows} rows, {sum(d['tokens'] for d in docs)} tokens from {len(chunks)} chunks -> {a.final}")
