#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""traces226.py SESSDOCS_DEDUP_DIR TOKENIZER.json OUT_PREFIX --tr32k DIR [--tr32k DIR ..] [--min 120000] [--n 12] [--kinds main,worker] -- the deep trace
corpus of w226 (source 2): our own pi-session documents (xcorp/sessdocs.py + sessdedup.py output: whole session pieces of <= 250k tokens, cut at turn
boundaries, redacted) at full length, for a bf16 golden (radiance --kld-record, TP4) and QAD's front streams.
Selection: pieces of --kinds with >= --min tokens; a piece is dropped when it holds a tr32k document already in a QAD plan (probes: three 32-token
windows of every tr32k document, at 1/4, 1/2 and 3/4 of it, looked up in the piece's 32-gram hashes; a hit = at least 2 of one document's 3 probes,
so a shared system prompt / tool block alone is not one); then at most one piece a session, the
longest sessions' pieces first, alternating kinds, until --n pieces.
Writes OUT_PREFIX.jsonl (radiance --kld-corpus lines {prompt, source, score_from: 0}) and OUT_PREFIX.jsonl.meta.jsonl ({doc, piece, source, category,
kind, session, tokens, assistant_tokens}). PRIVATE: the user's traces; the output stays on our machines (VM, box2), never in the repo."""
import argparse, glob, json, os
import numpy as np
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("tok"); ap.add_argument("out")
ap.add_argument("--tr32k", action="append", default=[]); ap.add_argument("--min", type=int, default=120000)
ap.add_argument("--n", type=int, default=12); ap.add_argument("--kinds", default="main,worker")
a = ap.parse_args()
W = 32
tk = Tokenizer.from_file(a.tok)
MUL = np.uint64(1099511628211)


def gram_hash(ids):
    """32-gram hashes (FNV-style over int64 token ids; wraps mod 2^64)"""
    x = np.asarray(ids, np.uint64)
    if len(x) < W: return np.zeros(0, np.uint64)
    h = np.full(len(x) - W + 1, np.uint64(14695981039346656037))
    with np.errstate(over="ignore"):
        for k in range(W): h = (h ^ x[k:len(x) - W + 1 + k]) * MUL
    return h


probes, owner = [], []
for d in a.tr32k:
    m = json.load(open(f"{d}/kld.json"))["docs"]; t = np.fromfile(f"{d}/tokens.i32", np.int32); off = 0
    for doc in m:
        n = doc["tokens"]; ids = t[off:off + n]; off += n
        for f in (0.25, 0.5, 0.75):
            s = int(n * f)
            if s + W <= n: probes.append(gram_hash(ids[s:s + W])[0]); owner.append(f"{os.path.basename(d)}:{doc['row0']}")
probes = np.array(probes, np.uint64); owner = np.array(owner); print(f"{len(probes)} probes from {len(set(owner))} tr32k documents")


def holds_tr32k(ids):
    h = np.isin(probes, gram_hash(ids))
    if not h.any(): return False
    _, c = np.unique(owner[h], return_counts=True); return bool((c >= 2).any())
kinds = a.kinds.split(",")
cands = []
for f in sorted(glob.glob(f"{a.src}/*.jsonl")):
    for ln in open(f):
        d = json.loads(ln)
        if d["kind"] in kinds and d["tokens"] >= a.min: cands.append(d)
print(f"{len(cands)} candidate pieces (kinds {kinds}, >= {a.min} tokens)")
cands.sort(key=lambda d: -d["tokens"])
keep, seen, hit = [], set(), 0
by_kind = {k: [d for d in cands if d["kind"] == k] for k in kinds}
ptr = {k: 0 for k in kinds}
while len(keep) < a.n and any(ptr[k] < len(by_kind[k]) for k in kinds):
    for k in kinds:
        while ptr[k] < len(by_kind[k]):
            d = by_kind[k][ptr[k]]; ptr[k] += 1
            if d["session"] in seen: continue
            ids = tk.encode(d["prompt"], add_special_tokens=False).ids
            if holds_tr32k(ids): hit += 1; continue
            d["_ntok"] = len(ids); keep.append(d); seen.add(d["session"]); break
        if len(keep) >= a.n: break
print(f"kept {len(keep)} pieces ({hit} dropped: content in tr32k)")
with open(a.out + ".jsonl.part", "w") as fo, open(a.out + ".jsonl.meta.jsonl.part", "w") as fm:
    for i, d in enumerate(keep):
        fo.write(json.dumps({"prompt": d["prompt"], "source": "pi-sessions-deep", "score_from": 0}) + "\n")
        fm.write(json.dumps({"doc": i, "piece": d["piece"], "source": "pi-sessions-deep", "category": "agentic", "kind": d["kind"], "session": d["session"],
                             "tokens": d["_ntok"], "assistant_tokens": d.get("assistant_tokens")}) + "\n")
        print(f"  {i:2d} {d['kind']:6s} tokens {d['_ntok']:7d} assistant {d.get('assistant_tokens')} piece {d['piece']}")
os.replace(a.out + ".jsonl.part", a.out + ".jsonl"); os.replace(a.out + ".jsonl.meta.jsonl.part", a.out + ".jsonl.meta.jsonl")
