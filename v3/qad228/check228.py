#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check228.py GOLDEN CORPUS TOKENIZER -- w228 step 5 checks of the on-policy golden (radiance --kld-record of docs228.py's corpus):
rows = the sum of the meta's gen_rows (every row a generated token), each document's tokens / score_from as docs228.py counted them,
tokens.i32 = the tokenizer's encoding of the corpus prompt (what docs228.py's gen boundary was computed on), gen.u8 all ones, the stored
top-512 distributions sane (sorted, total mass <= 1, the true token's nll finite). Prints one summary line; exit 1 on a failed check."""
import json, sys

import numpy as np
from transformers import AutoTokenizer

g, corp, tkp = sys.argv[1:4]
m = json.load(open(f"{g}/kld.json")); docs = m["docs"]; R = m["rows"]; K = m["top"]
meta = [json.loads(l) for l in open(f"{g}/corpus.meta.jsonl")]
lines = open(corp).read().rstrip("\n").split("\n")
tok = np.memmap(f"{g}/tokens.i32", np.int32, "r"); gen = np.fromfile(f"{g}/gen.u8", np.uint8)
rows = np.memmap(f"{g}/rows.f32", np.float32, "r", shape=(R, 3))
top = np.memmap(f"{g}/top.bin", np.dtype([("id", "<i4"), ("lp", "<f4")]), "r", shape=(R, K))
tk = AutoTokenizer.from_pretrained(tkp)
bad = []
if len(docs) != len(meta) or len(docs) != len(lines): bad.append(f"doc counts golden {len(docs)} meta {len(meta)} corpus {len(lines)}")
if R != sum(x["gen_rows"] for x in meta): bad.append(f"rows {R} != meta gen rows {sum(x['gen_rows'] for x in meta)}")
if len(gen) != R or not gen.all(): bad.append("gen.u8 not all ones / wrong length")
off = 0; mism = 0
for i, (d, x, l) in enumerate(zip(docs, meta, lines)):
    if d["tokens"] != x["tokens"] or d["score_from"] != x["score_from"]: bad.append(f"doc {i}: tokens / score_from {d} vs {x}"); break
    ids = tk(json.loads(l)["prompt"], add_special_tokens=False)["input_ids"]
    if ids != tok[off:off + d["tokens"]].tolist(): mism += 1
    off += d["tokens"]
if mism: bad.append(f"{mism} documents whose stored tokens differ from the tokenizer's")
rng = np.random.default_rng(0); s = np.sort(rng.choice(R, min(R, 20000), replace=False))
t = top[s]; lp = t["lp"]
mass = np.exp(lp.astype(np.float64)).sum(1)
if not (np.diff(lp, axis=1) <= 1e-6).all(): bad.append("top lists not sorted")
if not (mass <= 1 + 1e-3).all(): bad.append(f"top mass > 1: max {mass.max()}")
nll = rows[s, 0]
if not np.isfinite(nll).all(): bad.append("non-finite nll")
print(f"check228: docs {len(docs)} rows {R} (= gen rows) tokens {off} top-{K} mass mean {mass.mean():.4f} nll mean {nll.mean():.4f} "
      f"sampled gen tokens {sum(x['gen_sampled'] for x in meta)}: {'OK' if not bad else 'FAIL ' + '; '.join(bad)}")
sys.exit(1 if bad else 0)
