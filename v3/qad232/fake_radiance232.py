#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fake_radiance232.py --config YAML -- stands in for radiance --kld-record in chaintest232.py: reads the yaml (kld-record, kld-corpus, tp, max-model-len), exits 3 when
tp == 2 and max-model-len > FAKE_TP2_MAX_MML (a deep chunk that does not fit TP2), else writes a fake golden (real tokens, valid random top-512 lists, score_from 0)."""
import json, os, sys
import numpy as np
from transformers import AutoTokenizer
y = {}
for l in open(sys.argv[sys.argv.index("--config") + 1]):
    if ":" in l and not l.strip().endswith(":"): k, v = l.strip().split(":", 1); y[k] = v.strip()
if int(y["tp"]) == 2 and int(y["max-model-len"]) > int(os.environ["FAKE_TP2_MAX_MML"]): print("fake: does not fit TP2"); sys.exit(3)
g = y["kld-record"]; os.makedirs(g); tk = AutoTokenizer.from_pretrained(os.environ["FAKE_TOK"]); rng = np.random.default_rng(0)
T, dd, r = [], [], 0
for l in open(y["kld-corpus"]).read().rstrip("\n").split("\n"):
    j = json.loads(l); ids = tk(j["prompt"], add_special_tokens=False)["input_ids"]; T += ids; dd.append({"source": j["source"], "tokens": len(ids), "score_from": 0, "row0": r}); r += len(ids) - 1
np.array(T, np.int32).tofile(f"{g}/tokens.i32")
p = 0.5 ** np.arange(512); lp = np.log(0.9 * p / p.sum()).astype(np.float32)
rec = np.zeros((r, 512), np.dtype([("id", "<i4"), ("lp", "<f4")])); rec["id"] = rng.integers(0, 248320, (r, 512)); rec["lp"] = lp; rec.tofile(f"{g}/top.bin")
np.stack([rng.random(r) * 3, rng.random(r), rng.random(r)], 1).astype(np.float32).tofile(f"{g}/rows.f32")
json.dump({"format": "radiance-kld-top-1", "model": "fake", "corpus": y["kld-corpus"], "corpus_fnv1a64": "0", "created": "now", "n_vocab": 248320, "top": 512, "rows": r, "docs": dd}, open(f"{g}/kld.json", "w"))
