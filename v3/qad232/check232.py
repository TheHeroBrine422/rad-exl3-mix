#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check232.py GOLDEN MASK CORPUS TOKENIZER -- w232 checks of the merged xg-fin golden: docs / rows against the meta (rows = tokens - 1 a doc, score_from 0),
gen mask = the meta's gen_rows per document and = the mask file, kld.json corpus hash = the corpus file's FNV-1a (radiance's), tokens.i32 = the tokenizer's
encoding of every corpus prompt, top-512 lists sane (sorted, mass <= 1, finite nll) on a 20,000-row sample. One summary line; exit 1 on a failed check."""
import json, sys

import numpy as np
from transformers import AutoTokenizer

g, maskf, corp, tkp = sys.argv[1:5]
m = json.load(open(f"{g}/kld.json")); docs = m["docs"]; R = m["rows"]; K = m["top"]
meta = [json.loads(l) for l in open(f"{g}/corpus.meta.jsonl")]
lines = open(corp).read().rstrip("\n").split("\n")
tok = np.memmap(f"{g}/tokens.i32", np.int32, "r"); gen = np.fromfile(f"{g}/gen.u8", np.uint8); mask = np.fromfile(maskf, np.uint8)
rows = np.memmap(f"{g}/rows.f32", np.float32, "r", shape=(R, 3))
top = np.memmap(f"{g}/top.bin", np.dtype([("id", "<i4"), ("lp", "<f4")]), "r", shape=(R, K))
tk = AutoTokenizer.from_pretrained(tkp)
bad = []
h = 1469598103934665603
for l in open(corp, "rb").read().split(b"\n")[:len(lines)]:
    for c in l: h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
if m.get("corpus_fnv1a64") != f"{h:016x}": bad.append(f"corpus hash {m.get('corpus_fnv1a64')} vs {h:016x}")
if len(docs) != len(meta) or len(docs) != len(lines): bad.append(f"doc counts golden {len(docs)} meta {len(meta)} corpus {len(lines)}")
if R != sum(x["tokens"] - 1 for x in meta): bad.append(f"rows {R} != sum(tokens - 1) {sum(x['tokens'] - 1 for x in meta)}")
if len(gen) != R or not np.array_equal(gen, mask): bad.append("gen.u8 != the mask file / wrong length")
if int(gen.sum()) != sum(x["gen_rows"] for x in meta): bad.append(f"gen rows {int(gen.sum())} != meta {sum(x['gen_rows'] for x in meta)}")
off = 0; mism = 0; gbad = 0
for i, (d, x, l) in enumerate(zip(docs, meta, lines)):
    if d["tokens"] != x["tokens"] or d["score_from"] != 0 or d["row0"] != off - i: bad.append(f"doc {i}: {d} vs {x}"); break
    ids = tk(json.loads(l)["prompt"], add_special_tokens=False)["input_ids"]
    if ids != tok[off:off + d["tokens"]].tolist(): mism += 1
    if int(gen[d["row0"]:d["row0"] + d["tokens"] - 1].sum()) != x["gen_rows"]: gbad += 1
    off += d["tokens"]
if mism: bad.append(f"{mism} documents whose stored tokens differ from the tokenizer's")
if gbad: bad.append(f"{gbad} documents whose gen rows differ from the meta")
rng = np.random.default_rng(0); s = np.sort(rng.choice(R, min(R, 20000), replace=False))
t = top[s]; lp = t["lp"]; mass = np.exp(lp.astype(np.float64)).sum(1)
if not (np.diff(lp, axis=1) <= 1e-6).all(): bad.append("top lists not sorted")
if not (mass <= 1 + 1e-3).all(): bad.append(f"top mass > 1: max {mass.max()}")
nll = rows[s, 0]
if not np.isfinite(nll).all(): bad.append("non-finite nll")
print(f"check232: docs {len(docs)} rows {R} gen rows {int(gen.sum())} tokens {off} top-{K} mass mean {mass.mean():.4f} nll mean {nll.mean():.4f}: {'OK' if not bad else 'FAIL ' + '; '.join(bad)}")
sys.exit(1 if bad else 0)
