#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sel229.py XGAP_DIR TOKENIZER_DIR OUT_DIR [--seed 229] -- lead 11-25 (user: "test xgap"): the first xgap screen's documents.
  OUT/xg-judge.jsonl: the gap judge, xgap-v1 HOLDOUT docs only, per-cell token targets (JUDGE below), each doc cut to <= 65,536 tokens.
  OUT/xg-scr1.jsonl : screen training docs, xgap-v1 NON-holdout docs only, per-cell token targets (SCREEN below), each doc cut to
                      <= 8,600 tokens (the trainer caps at 8,192); radiance's kld-corpus has no maxtok, so the prompt text is cut at
                      the character offset of token CAP (fast-tokenizer offsets).
  Lines: {"source", "category", "kind", "prompt", "xg_id", "cell", "tokens_cut", "tokens_orig"}; OUT/<name>.meta.jsonl the same
  without prompt. Docs are drawn in a seeded order, round-robin over sources inside a cell, until the cell's target."""
import collections, json, os, random, sys
from transformers import AutoTokenizer

X, TK, OUT = sys.argv[1:4]
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 229
JUDGE = {"long-docs": 120_000, "agent-sessions": 220_000, "multilingual-text": 40_000, "multilingual-chat": 60_000,
         "long-chat": 70_000, "code-langs": 40_000, "tools-other": 50_000}
SCREEN = {"long-docs": 330_000, "agent-sessions": 1_600_000, "multilingual-text": 330_000, "multilingual-chat": 440_000,
          "long-chat": 380_000, "code-langs": 220_000, "tools-other": 380_000}
tok = AutoTokenizer.from_pretrained(TK)
man = [json.loads(l) for l in open(f"{X}/manifest.jsonl")]
xg = [m for m in man if m["origin"] == "xgap-v1"]
lines = {}
def text(m):
    f = m["file"]
    if f not in lines: lines[f] = open(f"{X}/{f}").read().rstrip("\n").split("\n")
    return json.loads(lines[f][m["line"]])
def cut(prompt, cap):
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    if n <= cap: return prompt, n, n
    return prompt[:enc["offset_mapping"][cap][0]], cap, n
os.makedirs(OUT, exist_ok=True)
for name, hold, targets, cap in (("xg-judge", True, JUDGE, 65_536), ("xg-scr1", False, SCREEN, 8_600)):
    rng = random.Random(f"{SEED}-{name}")
    out, meta, got = [], [], collections.Counter()
    for cell, tgt in targets.items():
        by = collections.defaultdict(list)
        for m in xg:
            if m["cell"] == cell and bool(m.get("holdout")) == hold: by[m["source"]].append(m)
        for v in by.values(): rng.shuffle(v)
        srcs = sorted(by); rng.shuffle(srcs); k = 0
        while got[cell] < tgt and any(by[s] for s in srcs):
            s = srcs[k % len(srcs)]; k += 1
            if not by[s]: continue
            m = by[s].pop(); j = text(m)
            p, nc, no = cut(j["prompt"], cap)
            if nc < 64: continue
            got[cell] += nc
            d = {"source": j.get("source", m["source"]), "category": j.get("category", cell), "kind": j.get("kind", m.get("kind")),
                 "xg_id": m["id"], "cell": cell, "tokens_cut": nc, "tokens_orig": no}
            meta.append(d); out.append(dict(d, prompt=p))
    with open(f"{OUT}/{name}.jsonl.part", "w") as f:
        for d in out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(f"{OUT}/{name}.jsonl.part", f"{OUT}/{name}.jsonl")
    with open(f"{OUT}/{name}.meta.jsonl", "w") as f:
        for d in meta: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(name, len(out), "docs", dict(got), "total", sum(got.values()))
