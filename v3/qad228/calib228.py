#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""calib228.py CALIB_GOLDEN CALIB_CORPUS POOLS_V3 OUT --kind chat|text --exclude-golden J217_DIR --exclude-ids JLONG.json
--exclude-judge CONTPOOL.json [--seed 228] [--max-p 8192] -- w228 second batch (lead 11-24): the on-policy cut points of the calib corpus,
in cuts228.py's layout, so gen228.py and docs228.py run on it unchanged.
calib.jsonl has two kinds (field "kind"): chat (template text) and text (raw documents). --kind chat: cuts at assistant turns as below
(the generation prompt + up to 4,096 new tokens); --kind text: one cut a document at a line / paragraph boundary near a seeded 30-70 %
point of its first min(length, 8,571) tokens (prefix <= 6,000), the model continuing the raw text (`raw`: no chat template, no
generation prompt) for up to 2,048 new tokens: the boundary is the paragraph end (a token ending in two newlines) nearest the drawn
point inside 30-70 %, else the nearest line end (a token ending in a newline), else a newline followed only by indentation; documents
with none are skipped.

Chat documents: calib ids whose pools-v3 pool is qad / fit / search, minus j217's documents (`orig` in the j217 golden's kld.json), golden-jlong's
(jlong-ids.json) and the jc228 judge (contpool.json "judge"); their tokens are the calib golden's tokens.i32 (the official tokenisation).
Candidate cut: an assistant turn start (`<|im_start|>assistant`) right after a user-role turn (a user message or a tool response), with
prefix p (tokens before that <|im_start|>) 1 <= p <= --max-p. Pick (seeded): rounds; each round takes one not-yet-picked candidate at
random from every document that has one left, documents in a seeded order interleaved by category (round robin); the generator walks
this order and stops at its token target, so the batch spreads over documents and categories first, then deepens.
Writes OUT/sess/<calib id>.npz (tokens, starts, cands), OUT/cuts.jsonl ({cut, sidx = calib id, group = cut, file: "calib:<id>", kind =
category, p, depth = round, ncand, header = the system turn's end or 0, follow: false, raw, max_new, batch: calib-chat | calib-text}),
OUT/cuts.stats.json. Text documents: the same pool / exclusion filter, one round, categories round robin.
The calib corpus is the challenge's: the cut file holds ids and offsets only."""
import argparse, collections, json, os, random

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("golden"); ap.add_argument("corpus"); ap.add_argument("pools"); ap.add_argument("out")
ap.add_argument("--exclude-golden", action="append", default=[]); ap.add_argument("--exclude-ids", action="append", default=[])
ap.add_argument("--exclude-judge", action="append", default=[])
ap.add_argument("--tok", default="/data/q27/models/qwen38-27b-bf16")
ap.add_argument("--kind", choices=["chat", "text"], required=True)
ap.add_argument("--seed", type=int, default=228); ap.add_argument("--max-p", type=int, default=8192)
a = ap.parse_args()
S, EN, A, U = 248045, 248046, 74455, 846          # <|im_start|> <|im_end|> assistant user (Qwen3.8 tokenizer ids)
m = json.load(open(f"{a.golden}/kld.json")); docs = m["docs"]
tok = np.memmap(f"{a.golden}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in docs])])
pool = json.load(open(a.pools))["pool"]
CL = [json.loads(l) for l in open(a.corpus)]
cat = [x.get("category", "?") for x in CL]; kind = [x["kind"] for x in CL]
assert len(cat) == len(docs) == len(pool)
TEXT_MAX_P, TEXT_NEW, CHAT_NEW = 6000, 2048, 4096
if a.kind == "text":
    from transformers import AutoTokenizer
    tks = AutoTokenizer.from_pretrained(a.tok).convert_ids_to_tokens(list(range(248320)))
    vocab = np.array([t or "" for t in tks], dtype=object)
    is_para = np.array([t.endswith("ĊĊ") for t in vocab]); is_line = np.array([t.endswith("Ċ") for t in vocab])
    is_ind = np.array(["Ċ" in t and t.rstrip("Ġĉ").endswith("Ċ") for t in vocab])
excl = collections.Counter(); ex = set()
for g in a.exclude_golden:
    s = {d["orig"] for d in json.load(open(f"{g}/kld.json"))["docs"]}; ex |= s; excl[os.path.basename(g.rstrip("/"))] = len(s)
for f in a.exclude_ids:
    s = set(json.load(open(f))); ex |= s; excl[os.path.basename(f)] = len(s)
for f in a.exclude_judge:
    s = set(json.load(open(f))["judge"]); ex |= s; excl[os.path.basename(f) + ":judge"] = len(s)
os.makedirs(f"{a.out}/sess", exist_ok=True)
st = collections.Counter(); C = {}
rng = random.Random(a.seed)
for i, d in enumerate(docs):
    if kind[i] != a.kind: continue
    if pool[i] not in ("qad", "fit", "search"): st["skip_pool_" + pool[i]] += 1; continue
    if i in ex: st["skip_excluded"] += 1; continue
    t = np.asarray(tok[off[i]:off[i + 1]])
    if a.kind == "chat":
        st_ = np.where(t == S)[0]
        c = [int(p) for k, p in enumerate(st_) if k > 0 and 1 <= p <= a.max_p and p + 1 < len(t) and t[p + 1] == A
             and t[st_[k - 1] + 1] == U and p >= 2 and t[p - 2] == EN]
        if not c: st["no_cut"] += 1; continue
        header = int(st_[1]) if len(st_) > 1 and st_[0] == 0 else 0
    else:
        L = min(len(t), int(TEXT_MAX_P / 0.7)); lo, hi = int(0.3 * L), int(0.7 * L); x = rng.uniform(0.3, 0.7) * L
        if len(t) < 64 or hi <= max(lo, 1): st["text_too_short"] += 1; continue
        ks = np.arange(max(lo, 1), hi + 1)                       # cut before token k: token k - 1 ends the line
        c = []
        for m_ in (is_para, is_line, is_ind):
            ok = ks[m_[t[ks - 1]]]
            if len(ok): c = [int(ok[np.argmin(np.abs(ok - x))])]; st["text_cut_" + ("para" if m_ is is_para else "line" if m_ is is_line else "indent")] += 1; break
        if not c: st["text_no_boundary"] += 1; continue
        header = 0; st_ = np.array([0], np.int64)
    np.savez(f"{a.out}/sess/{i}.npz", tokens=t[:max(c) + 64], starts=st_[st_ < max(c) + 64].astype(np.int32), cands=np.asarray(c, np.int32))
    C[i] = (c, header)
st["docs"] = len(C); st["candidates"] = sum(len(c) for c, _ in C.values())
by_cat = collections.defaultdict(list)
for i in sorted(C): by_cat[cat[i]].append(i)
for k in sorted(by_cat): rng.shuffle(by_cat[k])
order = []
cats = sorted(by_cat); ptr = {k: 0 for k in cats}
while any(ptr[k] < len(by_cat[k]) for k in cats):
    for k in cats:
        if ptr[k] < len(by_cat[k]): order.append(by_cat[k][ptr[k]]); ptr[k] += 1
left = {i: list(C[i][0]) for i in order}
rows, rnd = [], 0
while any(left.values()):
    for i in order:
        if not left[i]: continue
        p = left[i].pop(rng.randrange(len(left[i])))
        rows.append({"cut": len(rows), "sidx": i, "group": len(rows), "file": f"calib:{i}", "kind": cat[i], "p": p, "depth": rnd,
                     "ncand": len(C[i][0]), "header": C[i][1], "follow": False, "raw": a.kind == "text",
                     "max_new": TEXT_NEW if a.kind == "text" else CHAT_NEW, "batch": f"calib-{a.kind}"})
    rnd += 1
st["cuts"] = len(rows); st["rounds"] = rnd; st["excluded_lists"] = dict(excl); st["excluded_union"] = len(ex)
st["docs_by_category"] = dict(collections.Counter(cat[i] for i in C))
st["docs_by_pool"] = dict(collections.Counter(pool[i] for i in C))
with open(f"{a.out}/cuts.jsonl.part", "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
os.replace(f"{a.out}/cuts.jsonl.part", f"{a.out}/cuts.jsonl")
json.dump(st, open(f"{a.out}/cuts.stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
