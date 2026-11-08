#!/usr/bin/env python3
"""sessdedup.py [--dir DIR] [--near J] [--min-words N] -- stage 2: dedup sessdocs.py's documents at turn level,
exact and fuzzy, across every document (user 2026-11-07). Rebuilds DIR/dedup/ from DIR/docs/ each run (~1-2 min).

Units: a document = header (system + tools) + turns; a turn starts at a real user message (a `<|im_start|>user`
block that is not a `<tool_response>`) and runs to the next one, so tool calls stay with their responses.
Order: documents by session start time (the timestamp in the file path), oldest first; the first copy is kept.
  1. attachments  `<file name=...>` blocks in user turns (print-mode runs attach the same files again and again):
                  the first copy of a content is kept, later copies become `[contents as attached earlier]`
  2. exact        a turn whose text (after 1) was seen before
  3. near         a turn of >= --min-words words whose estimated Jaccard similarity (bottom-k MinHash, k 128, word
                  5-gram shingles) with an earlier turn is >= --near (0.8)
Action for a duplicate turn (2 or 3):
  - leading duplicates (the turns before the document's first new one: continued / forked sessions, re-runs) are
    dropped; a document with no new turn is dropped;
  - duplicates after a new turn stay (deleting them would break the context later turns rely on) and are listed in
    `dup_spans` [[char start, char end, "exact" | "near"]] of the output prompt, for loss masking.
Output: DIR/dedup/*.jsonl (the docs' fields + dup_spans, tokens recounted, masked_tokens, masked_assistant_tokens),
DIR/dedup-stats.json by kind: tokens and assistant tokens before / after / masked, turn counts, attachment hits.
Unique trainable tokens = tokens - masked_tokens (assistant: assistant_tokens - masked_assistant_tokens).
Run: /root/sessvenv/bin/python -I xcorp/sessdedup.py"""
import argparse, collections, glob, hashlib, json, os, re, shutil, sys, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "..", "local", "sessdocs"))
ap.add_argument("--near", type=float, default=0.8); ap.add_argument("--min-words", type=int, default=64)
ap.add_argument("--k", type=int, default=128)
a = ap.parse_args()
DIR = os.path.abspath(a.dir)
import numpy as np
from transformers import AutoTokenizer
from transformers.utils import logging as hflog
hflog.set_verbosity_error()
tok = AutoTokenizer.from_pretrained(os.environ.get("XCORP_TOKENIZER", "/workspace/models/qwen38-27b-bf16"))
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0

FILE = re.compile(r'<file name="([^"]*)">\n?(.*?)</file>', re.S)
TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z")


def split(prompt):
    """-> header, [turn texts]; each turn begins with '<|im_start|>user' (a real user message)"""
    segs = prompt.split("<|im_start|>"); head, turns = segs[0], []
    for s in segs[1:]:
        s = "<|im_start|>" + s
        if s.startswith("<|im_start|>user") and not s.startswith("<|im_start|>user\n<tool_response>"): turns.append(s)
        elif turns: turns[-1] += s
        else: head += s
    return head, turns


def assistant_tokens(text):
    return sum(n_tok("<|im_start|>" + s) for s in text.split("<|im_start|>")[1:] if s.startswith("assistant"))


def sketch(text):
    w = re.findall(r"\w+", text.lower())
    if len(w) < a.min_words: return None
    ids = np.fromiter((zlib.crc32(x.encode()) for x in w), np.uint64, len(w))
    h = np.zeros(len(w) - 4, np.uint64)
    for j in range(5): h = h * np.uint64(1099511628211) ^ ids[j:len(w) - 4 + j]
    return np.unique(h * np.uint64(0x9E3779B97F4A7C15))[:a.k]


t0 = time.time()
docs = []
for f in sorted(glob.glob(f"{DIR}/docs/*.jsonl")):
    for l in open(f): d = json.loads(l); d["_out"] = os.path.basename(f); docs.append(d)
docs.sort(key=lambda d: ((TS.search(d["file"]) or [""])[0], d["file"], d["piece"]))

seen_att, seen_turn = set(), set()
sk_store, inv = [], collections.defaultdict(list)
tot = collections.defaultdict(collections.Counter)
out = collections.defaultdict(list)


def near_of(sk):
    if sk is None: return False
    cnt = collections.Counter()
    for v in sk.tolist():
        for j in inv.get(v, ()): cnt[j] += 1
    for j, n in cnt.most_common(10):
        if n < 4: break
        o = sk_store[j]; u = np.union1d(sk, o)[:a.k]
        if (np.isin(u, sk) & np.isin(u, o)).mean() >= a.near: return True
    return False


for d in docs:
    T = tot[d["kind"]]; T["docs_in"] += 1; T["tokens_in"] += d["tokens"]; T["assistant_in"] += d["assistant_tokens"]
    head, turns = split(d["prompt"])
    status = []
    for tx in turns:
        def rep(m):
            h = hashlib.sha256(m.group(2).encode()).hexdigest()
            if h in seen_att: T["attachments_dup"] += 1; return f'<file name="{m.group(1)}">[contents as attached earlier]</file>'
            seen_att.add(h); return m.group(0)
        tx = FILE.sub(rep, tx)
        h = hashlib.sha256(tx.encode()).hexdigest(); sk = sketch(tx)
        if h in seen_turn: kind = "exact"
        elif near_of(sk): kind = "near"
        else: kind = None
        seen_turn.add(h)
        if sk is not None:
            sk_store.append(sk)
            for v in sk.tolist():
                if len(inv[v]) < 500: inv[v].append(len(sk_store) - 1)
        status.append((tx, kind))
    T["turns"] += len(status)
    first_new = next((i for i, (_, k) in enumerate(status) if k is None), None)
    if first_new is None: T["docs_dropped"] += 1; T["turns_dropped"] += len(status); continue
    T["turns_dropped"] += first_new
    prompt, spans, masked, masked_a = head, [], 0, 0
    for tx, k in status[first_new:]:
        if k:
            spans.append([len(prompt), len(prompt) + len(tx), k]); T[f"turns_masked_{k}"] += 1
            masked += n_tok(tx); masked_a += assistant_tokens(tx)
        prompt += tx
    e = {x: v for x, v in d.items() if x != "_out"}
    e.update(prompt=prompt, dup_spans=spans, tokens=n_tok(prompt), assistant_tokens=assistant_tokens(prompt),
             masked_tokens=masked, masked_assistant_tokens=masked_a)
    out[d["_out"]].append(e)
    T["docs_out"] += 1; T["tokens_out"] += e["tokens"]; T["assistant_out"] += e["assistant_tokens"]
    T["masked_tokens"] += masked; T["masked_assistant_tokens"] += masked_a

tmp = f"{DIR}/dedup.part"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
for name, es in out.items():
    with open(f"{tmp}/{name}", "w") as fo:
        for e in sorted(es, key=lambda e: e["piece"]): fo.write(json.dumps(e, ensure_ascii=False) + "\n")
shutil.rmtree(f"{DIR}/dedup", ignore_errors=True); os.replace(tmp, f"{DIR}/dedup")
allc = sum(tot.values(), collections.Counter())
json.dump({"updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "near": a.near, "min_words": a.min_words,
           "by_kind": {k: dict(v) for k, v in sorted(tot.items())}, "all": dict(allc)},
          open(f"{DIR}/dedup-stats.json", "w"), indent=1)
print(f"dedup in {time.time() - t0:.0f} s  (M tokens: in -> out, masked; assistant in -> out, masked)")
for k, v in sorted(tot.items()) + [("ALL", allc)]:
    print(f"{k:7s} docs {v['docs_in']:4d}->{v['docs_out']:4d}  tokens {v['tokens_in']/1e6:6.2f} -> {v['tokens_out']/1e6:6.2f}"
          f" masked {v['masked_tokens']/1e6:5.2f}  assistant {v['assistant_in']/1e6:5.2f} -> {v['assistant_out']/1e6:5.2f}"
          f" masked {v['masked_assistant_tokens']/1e6:5.2f}  turns {v['turns']} dropped {v['turns_dropped']}"
          f" masked exact {v['turns_masked_exact']} near {v['turns_masked_near']}  att-dup {v['attachments_dup']}")
