#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""merge232.py DOCSDIR RECDIR FINAL CORPUS_OUT MASK_OUT TOKENIZER -- w232: merge the chunk goldens RECDIR/c<k> (radiance --kld-record of
DOCSDIR/xg-fin.c<k>.jsonl, listed in DOCSDIR/xg-fin.chunks.json) into ONE golden FINAL for the corpus DOCSDIR/xg-fin.jsonl (chunks concatenated in chunk
order). tokens.i32 / rows.f32 / top.bin are concatenated, kld.json is the first chunk's with docs / rows / corpus / corpus_fnv1a64 rewritten, the gen mask
is made by genmask181 on the merged golden (written to FINAL/gen.u8 and MASK_OUT), corpus.meta.jsonl = the merged meta, the corpus is copied to
CORPUS_OUT (+ .meta.jsonl). FINAL is written as FINAL.part and renamed. Each chunk is checked against its meta (docs, tokens, rows) first."""
import json, os, shutil, subprocess, sys

docs_dir, rec, final, corpus_out, mask_out, tkp = sys.argv[1:7]
here = os.path.dirname(os.path.abspath(__file__))
genmask = os.environ.get("W232_GENMASK", os.path.join(here, "..", "qad181", "genmask181.py"))


def fnv1a64(path):
    """radiance's corpus hash (core/kld.cpp): FNV-1a 64 chained over every getline() line without its newline, offset basis 1469598103934665603"""
    h = 1469598103934665603
    lines = open(path, "rb").read().split(b"\n")
    if lines and lines[-1] == b"": lines.pop()
    for l in lines:
        for c in l: h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


chunks = json.load(open(f"{docs_dir}/xg-fin.chunks.json")); corpus = f"{docs_dir}/xg-fin.jsonl"
meta_all = [json.loads(l) for l in open(corpus + ".meta.jsonl")]
tmp = final + ".part"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
docs, rows, base = [], 0, None
with open(f"{tmp}/tokens.i32", "wb") as ft, open(f"{tmp}/rows.f32", "wb") as fr, open(f"{tmp}/top.bin", "wb") as fp:
    for ch in chunks:
        g = f"{rec}/c{ch['k']}"; assert os.path.exists(f"{g}/.done"), f"chunk {g} not recorded"
        m = json.load(open(f"{g}/kld.json")); base = base or m
        cm = [json.loads(l) for l in open(f"{docs_dir}/{ch['file']}.meta.jsonl")]
        assert len(m["docs"]) == len(cm) == ch["docs"], (g, len(m["docs"]), len(cm), ch["docs"])
        for d, x in zip(m["docs"], cm): assert d["tokens"] == x["tokens"] and d["score_from"] == 0, (g, d, x)
        assert m["rows"] == ch["rows"] and m["top"] == base["top"] and m["n_vocab"] == base["n_vocab"], (g, m["rows"], ch["rows"])
        for d in m["docs"]:
            docs.append({"source": d["source"], "tokens": d["tokens"], "score_from": d["score_from"], "row0": rows + d["row0"], "orig": len(docs)})
        for fn, out in (("tokens.i32", ft), ("rows.f32", fr), ("top.bin", fp)):
            with open(f"{g}/{fn}", "rb") as f: shutil.copyfileobj(f, out, 1 << 24)
        rows += m["rows"]
assert len(docs) == len(meta_all), (len(docs), len(meta_all))
for i, (d, x) in enumerate(zip(docs, meta_all)): assert d["tokens"] == x["tokens"], (i, d, x)
os.makedirs(os.path.dirname(corpus_out), exist_ok=True)
shutil.copy(corpus, corpus_out + ".part"); os.replace(corpus_out + ".part", corpus_out)
shutil.copy(corpus + ".meta.jsonl", corpus_out + ".meta.jsonl.part"); os.replace(corpus_out + ".meta.jsonl.part", corpus_out + ".meta.jsonl")
k = dict(base); k["docs"] = docs; k["rows"] = rows; k["corpus"] = corpus_out; k["corpus_fnv1a64"] = f"{fnv1a64(corpus):016x}"
json.dump(k, open(f"{tmp}/kld.json", "w"))
shutil.copy(corpus + ".meta.jsonl", f"{tmp}/corpus.meta.jsonl")
r = subprocess.run([sys.executable, "-I", genmask, "make", tmp, mask_out, tkp], capture_output=True, text=True)
print(r.stdout.strip()); assert r.returncode == 0, r.stderr[-2000:]
shutil.copy(mask_out, f"{tmp}/gen.u8"); open(f"{tmp}/.done", "w").close()
shutil.rmtree(final, ignore_errors=True); os.replace(tmp, final)
print(f"merged xg-fin: {len(docs)} docs, {rows} rows, {sum(d['tokens'] for d in docs)} tokens from {len(chunks)} chunks -> {final}")
