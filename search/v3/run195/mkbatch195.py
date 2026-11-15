# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkbatch195.py XCORP_META TOKENIZER SUPPLY.jsonl OUT.jsonl [MIN_GEN=128] -- the fresh non-Latin xcorp batch (RULES-slices195 §1).
Every free non-Latin xcorp v1 document of supply195.py (used = []) whose first <= 8192 tokens carry >= MIN_GEN gen rows and
whose script is a writing system (not MATHEMATICAL), in a fixed shuffle (seed 195), as a radiance --kld-corpus:
lines {prompt, source, score_from: 0}; OUT.meta.jsonl per line {doc (xcorp v1 index), piece 0, source, category, session,
tokens, script, gen}. Pieces are cut and round-trip-checked exactly as xcorp/kldcorpus.py cuts the xc8k batches."""
import json, os, random, sys

import numpy as np
from transformers import AutoTokenizer

meta, tkp, sup, out = sys.argv[1:5]; MIN_GEN = int(sys.argv[5]) if len(sys.argv) > 5 else 128
tk = AutoTokenizer.from_pretrained(tkp)
docs = json.load(open(f"{meta}/kld.json"))["docs"]
T = np.memmap(f"{meta}/tokens.i32", np.int32, "r"); off = np.concatenate([[0], np.cumsum([d["tokens"] for d in docs])])
R = [json.loads(l) for l in open(sup)]
R = [r for r in R if not r["used"] and r["gen"] >= MIN_GEN and r["script"] != "MATHEMATICAL"]
random.Random(195).shuffle(R)
n = skipped = 0
with open(out + ".part", "w") as fo, open(out + ".meta.jsonl.part", "w") as fm:
    for r in R:
        i = r["doc"]; piece = np.asarray(T[off[i]:off[i + 1]])[:8192].tolist()
        text = tk.decode(piece); back = tk(text, add_special_tokens=False)["input_ids"]
        diff = abs(len(back) - len(piece)) + sum(x != y for x, y in zip(back, piece))
        if diff > 0.01 * len(piece): skipped += 1; continue
        fo.write(json.dumps({"prompt": text, "source": r["source"], "score_from": 0}) + "\n")
        fm.write(json.dumps({"doc": i, "piece": 0, "source": r["source"], "category": r["category"], "session": f"doc{i}",
                             "tokens": len(back), "script": r["script"], "gen": r["gen"]}) + "\n")
        n += len(back)
os.replace(out + ".part", out); os.replace(out + ".meta.jsonl.part", out + ".meta.jsonl")
print(f"MKBATCH195 DONE {n} tokens, {sum(1 for _ in open(out))} pieces, {skipped} skipped (round trip) -> {out}")
