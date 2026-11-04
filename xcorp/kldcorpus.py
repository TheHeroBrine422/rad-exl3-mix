#!/usr/bin/env python3
"""kldcorpus.py META OUT.jsonl CHUNK TOKENS SEED [--per-doc N] [--skip FILE] -- a --kld-corpus for radiance's
--kld-record from a tokenized corpus (xcorp/tokenize.py's META = kld.json docs + tokens.i32): documents in a fixed
shuffle (SEED), each cut into consecutive CHUNK-token pieces (at most N a document; default 1 = its first CHUNK
tokens, as the calib slices are cut), until TOKENS tokens. A piece is decoded back to text (radiance re-tokenizes
it; pieces whose round trip changes more than 1% of their tokens are skipped and counted). Output lines
{prompt, source, score_from: 0}; OUT.meta.jsonl beside it: per line the document index, piece number, source,
category, session (own traces: kld.json "session", else the document) for per-session bootstraps. --skip FILE: a
previous OUT.meta.jsonl whose documents are not used again (fresh slices).

--trace JSONL (META is then ignored, give "-"): agent traces as pisess.py renders them (one session piece a line,
Qwen's chat template, a system block first). Each piece is cut at message boundaries into chunks of <= CHUNK tokens,
every chunk = the piece's system block + consecutive messages starting at a user-role message (Qwen renders tool
results as user messages), so each looks like a real agent request; a single turn longer than the room left is cut
at CHUNK. Every chunk of a piece is used (--per-doc caps it). Session ids from the JSONL's "session".
Docs: docs/design/eval-data.md."""
import argparse, json, os, random
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("meta"); ap.add_argument("out"); ap.add_argument("chunk", type=int); ap.add_argument("tokens", type=int)
ap.add_argument("seed", type=int); ap.add_argument("--per-doc", type=int, default=1); ap.add_argument("--skip", action="append", default=[])
ap.add_argument("--trace", default=None)
ap.add_argument("--per-doc-trace", type=int, default=10**9)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(os.environ.get("XCORP_TOKENIZER", "/data/models/qwen38-27b-bf16"))
if a.trace:
    IM = tok.convert_tokens_to_ids("<|im_start|>")
    role_of = lambda ids, s: tok.decode([ids[s + 1]]).strip() if s + 1 < len(ids) else ""
    lines = [json.loads(l) for l in open(a.trace)]
    order = list(range(len(lines))); random.Random(a.seed).shuffle(order)
    used = {(r["doc"], r["piece"]) for f in a.skip for r in map(json.loads, open(f))}
    n = nch = 0
    with open(a.out + ".part", "w") as fo, open(a.out + ".meta.jsonl.part", "w") as fm:
        for i in order:
            d = lines[i]; ids = tok(d["prompt"], add_special_tokens=False)["input_ids"]
            st = [k for k, x in enumerate(ids) if x == IM] + [len(ids)]
            msgs = [(st[k], st[k + 1], role_of(ids, st[k])) for k in range(len(st) - 1)]
            ns = 0
            while ns < len(msgs) and msgs[ns][2] == "system": ns += 1
            sys_end = msgs[ns - 1][1] if ns else 0
            room = a.chunk - sys_end
            if room < 1024 or ns == len(msgs): continue
            k, p = ns, 0
            while k < len(msgs) and p < a.per_doc_trace:
                while k < len(msgs) and msgs[k][2] != "user": k += 1          # a chunk starts at a user message
                if k >= len(msgs): break
                a0 = msgs[k][0]; b = a0; k2 = k
                while k2 < len(msgs) and msgs[k2][1] - a0 <= room: b = msgs[k2][1]; k2 += 1
                if k2 == k: b = a0 + room; k2 = k + 1                         # one message longer than the room: cut it
                chunk = ids[:sys_end] + ids[a0:b]
                if (i, p) not in used:
                    fo.write(json.dumps({"prompt": tok.decode(chunk), "source": d.get("source", ""), "score_from": 0}) + "\n")
                    fm.write(json.dumps({"doc": i, "piece": p, "source": d.get("source", ""), "category": d.get("category", ""),
                                         "session": d.get("session", f"doc{i}"), "tokens": len(chunk), "sys": sys_end}) + "\n")
                    n += len(chunk); nch += 1
                p += 1; k = k2
                if n >= a.tokens: break
            if n >= a.tokens: break
    os.replace(a.out + ".part", a.out); os.replace(a.out + ".meta.jsonl.part", a.out + ".meta.jsonl")
    print(f"KLDCORPUS DONE (trace) {n} tokens, {nch} chunks -> {a.out}"); raise SystemExit
m = json.load(open(f"{a.meta}/kld.json")); docs = m["docs"]
T = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in docs])])
used = {json.loads(l)["doc"] for f in a.skip for l in open(f)}
order = [i for i in range(len(docs)) if i not in used]; random.Random(a.seed).shuffle(order)
n = skipped = 0
with open(a.out + ".part", "w") as fo, open(a.out + ".meta.jsonl.part", "w") as fm:
    for i in order:
        d = docs[i]; ids = np.asarray(T[off[i]:off[i + 1]])
        for p in range(min(a.per_doc, max(1, -(-len(ids) // a.chunk)))):
            piece = ids[p * a.chunk:(p + 1) * a.chunk]
            if len(piece) < 64: break
            text = tok.decode(piece.tolist())
            back = tok(text, add_special_tokens=False)["input_ids"]
            diff = abs(len(back) - len(piece)) + sum(x != y for x, y in zip(back, piece.tolist()))
            if diff > 0.01 * len(piece): skipped += 1; continue
            fo.write(json.dumps({"prompt": text, "source": d.get("source", ""), "score_from": 0}) + "\n")
            fm.write(json.dumps({"doc": i, "piece": p, "source": d.get("source", ""), "category": d.get("category", ""),
                                 "session": d.get("session", f"doc{i}"), "tokens": len(back)}) + "\n")
            n += len(back)
            if n >= a.tokens: break
        if n >= a.tokens: break
os.replace(a.out + ".part", a.out); os.replace(a.out + ".meta.jsonl.part", a.out + ".meta.jsonl")
print(f"KLDCORPUS DONE {n} tokens, {sum(1 for _ in open(a.out))} pieces, {skipped} skipped (round trip) -> {a.out}")
