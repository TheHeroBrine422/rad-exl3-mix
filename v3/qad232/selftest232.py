#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""selftest232.py TOKENIZER DOCSDIR WORKDIR [N_PER_CHUNK=6] -- no-GPU test of merge232.py + check232.py: takes the first N documents of two of sel232.py's chunks
(DOCSDIR/xg-fin.c*.jsonl), writes FAKE chunk goldens in radiance's --kld-record layout (real tokens; random but valid top-512 lists), merges them (merge232), checks the
result (check232: must print OK), then corrupts one stored token / one gen row and checks that check232 FAILS each time."""
import json, os, shutil, subprocess, sys

import numpy as np
from transformers import AutoTokenizer

tkp, docs, work = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 6
here = os.path.dirname(os.path.abspath(__file__)); PY = [sys.executable, "-I"]
shutil.rmtree(work, ignore_errors=True); os.makedirs(f"{work}/docs"); os.makedirs(f"{work}/rec")
tk = AutoTokenizer.from_pretrained(tkp); rng = np.random.default_rng(1)
allc = json.load(open(f"{docs}/xg-fin.chunks.json")); pick = [allc[-1], allc[-2]]
chunks, corpus, meta = [], [], []
for new_k, ch in enumerate(pick):
    L = open(f"{docs}/{ch['file']}").read().rstrip("\n").split("\n")[:N]; M = [json.loads(l) for l in open(f"{docs}/{ch['file']}.meta.jsonl")][:N]
    fn = f"xg-fin.c{new_k}.jsonl"; open(f"{work}/docs/{fn}", "w").write("\n".join(L) + "\n")
    open(f"{work}/docs/{fn}.meta.jsonl", "w").write("".join(json.dumps(m) + "\n" for m in M))
    chunks.append({"k": new_k, "file": fn, "docs": len(L), "tokens": sum(m["tokens"] for m in M), "max_tokens": max(m["tokens"] for m in M), "rows": sum(m["tokens"] - 1 for m in M),
                   "gen_rows": sum(m["gen_rows"] for m in M)})
    corpus += L; meta += M
    g = f"{work}/rec/c{new_k}"; os.makedirs(g); T = []; dd = []; r = 0
    for l, m in zip(L, M):
        ids = tk(json.loads(l)["prompt"], add_special_tokens=False)["input_ids"]; assert len(ids) == m["tokens"]
        T += ids; dd.append({"source": m["source"], "tokens": len(ids), "score_from": 0, "row0": r}); r += len(ids) - 1
    np.array(T, np.int32).tofile(f"{g}/tokens.i32")
    p = 0.5 ** np.arange(512); p = 0.9 * p / p.sum(); lp = np.log(p).astype(np.float32)
    rec = np.zeros((r, 512), np.dtype([("id", "<i4"), ("lp", "<f4")])); rec["id"] = rng.integers(0, 248320, (r, 512)); rec["lp"] = lp
    rec.tofile(f"{g}/top.bin"); np.stack([rng.random(r) * 3, rng.random(r), rng.random(r)], 1).astype(np.float32).tofile(f"{g}/rows.f32")
    json.dump({"format": "radiance-kld-top-1", "model": "fake", "corpus": "x", "corpus_fnv1a64": "0", "created": "now", "n_vocab": 248320, "top": 512, "rows": r, "docs": dd}, open(f"{g}/kld.json", "w"))
    open(f"{g}/.done", "w").close()
json.dump(chunks, open(f"{work}/docs/xg-fin.chunks.json", "w"))
open(f"{work}/docs/xg-fin.jsonl", "w").write("\n".join(corpus) + "\n")
open(f"{work}/docs/xg-fin.jsonl.meta.jsonl", "w").write("".join(json.dumps(m) + "\n" for m in meta))
F = f"{work}/final"; mask = f"{work}/xg-fin.u8"; co = f"{work}/out/xg-fin.jsonl"
r = subprocess.run(PY + [f"{here}/merge232.py", f"{work}/docs", f"{work}/rec", F, co, mask, tkp], capture_output=True, text=True); print(r.stdout.strip(), r.stderr[-1500:]); assert r.returncode == 0
def check(): return subprocess.run(PY + [f"{here}/check232.py", F, mask, co, tkp], capture_output=True, text=True)
c = check(); print(c.stdout.strip(), c.stderr[-800:]); assert c.returncode == 0 and "OK" in c.stdout
m = json.load(open(f"{F}/kld.json")); g = np.fromfile(f"{F}/gen.u8", np.uint8); assert int(g.sum()) == sum(x["gen_rows"] for x in meta) and m["rows"] == len(g)
t = np.memmap(f"{F}/tokens.i32", np.int32, "r+"); t[5] += 1; t.flush(); del t
c = check(); print("corrupt token ->", c.stdout.strip()[-160:]); assert c.returncode == 1 and "differ from the tokenizer" in c.stdout
t = np.memmap(f"{F}/tokens.i32", np.int32, "r+"); t[5] -= 1; t.flush(); del t
for fn in (f"{F}/gen.u8",):
    a = np.fromfile(fn, np.uint8); a[0] ^= 1; a.tofile(fn)
c = check(); print("corrupt gen ->", c.stdout.strip()[-160:]); assert c.returncode == 1
print("selftest232 OK")
