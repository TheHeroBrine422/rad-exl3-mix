#!/usr/bin/env python3
"""test_chunk.py CHUNKDIR TRACEDIR POOLS [TOKENIZER] -- independent checks of chunk.py's output (reads only its files, the
calib golden's trace cuts t<doc>.i32 / g<doc>.u8 and the pool file):
 1. each chunk's tokens = trace[:header] + trace[from:to] (tokens file and the re-tokenised prompt), <= max, score_from =
    header - 1, sha256s as in the manifest;
 2. cuts at message boundaries: a chunk body starts with <|im_start|>user and ends with <|im_end|> \\n (or the trace's
    end); tool-call / tool-response tags balanced inside every chunk body;
 3. a trace's chunks tile [header, n) exactly;
 4. pools: every slice holds only traces of its pool, reserved traces in no slice, agent-verify-whole = the routine
    verify traces' original lines;
 5. gen rows = the calib golden's gen rows of the same target tokens.
Exit 1 on any failure."""
import hashlib, json, sys
import numpy as np
from tokenizers import Tokenizer

cd, td, pools = sys.argv[1:4]
tk = Tokenizer.from_file(sys.argv[4] if len(sys.argv) > 4 else "/workspace/models/qwen38-27b-bf16/tokenizer.json")
pool = json.load(open(pools))["pool"]
man = json.load(open(f"{cd}/manifest.json"))
IMS, IME, NL, USER = (tk.token_to_id(x) for x in ("<|im_start|>", "<|im_end|>", "Ċ", "user"))
TAGS = [(tk.token_to_id(a), tk.token_to_id(b)) for a, b in (("<tool_call>", "</tool_call>"), ("<tool_response>", "</tool_response>"))]
fails = []


def ok(c, msg):
    if not c: fails.append(msg)


sha = lambda b: hashlib.sha256(b).hexdigest()
X = round(man["max_tokens"] / 1024)
want_pool = {f"agent-search{X}": "search", f"agent-verify{X}": "verify", f"qad{X}": "qad", f"verify-reserved{X}": "verify",
             "agent-verify-whole": "verify"}
tr = {t["doc"]: t for t in man["traces"]}
for name, s in man["slices"].items():
    lines = open(f"{cd}/{name}.jsonl").read().rstrip("\n").split("\n")
    toks = np.fromfile(f"{cd}/{name}.tokens.i32", np.int32); gen = np.fromfile(f"{cd}/{name}.gen.u8", np.uint8)
    ok(sha(open(f"{cd}/{name}.jsonl", "rb").read()) == s["sha256"], f"{name}: file sha256")
    ok(len(lines) == s["docs"] == len(s["members"]), f"{name}: doc count")
    to = go = 0
    for (doc, c), l in zip(s["members"], lines):
        j = json.loads(l); T = np.fromfile(f"{td}/t{doc}.i32", np.int32); G = np.fromfile(f"{td}/g{doc}.u8", np.uint8)
        ok(pool[doc] == want_pool[name], f"{name}: doc {doc} is in pool {pool[doc]}")
        ok(("reserved" in tr[doc]) == (name.startswith("verify-reserved")), f"{name}: doc {doc} reserved status")
        ct = np.array(tk.encode(j["prompt"], add_special_tokens=False).ids, np.int32)
        if c == "whole":
            ok((ct == T).all() and len(ct) == len(T), f"{name}: doc {doc} whole tokens"); sf, body0 = 0, 1
            ok(sha(l.encode()) == tr[doc]["whole"]["line_sha256"], f"{name}: doc {doc} line sha256")
        else:
            m = tr[doc]["chunks"][c]; H = tr[doc]["header_tokens"]; a, b = m["from"], m["to"]
            ok(len(ct) == H + b - a and (ct[:H] == T[:H]).all() and (ct[H:] == T[a:b]).all(), f"{name}: doc {doc} chunk {c} tokens")
            ok(len(ct) <= man["max_tokens"], f"{name}: doc {doc} chunk {c} longer than max")
            ok(j["score_from"] == H - 1 == m["score_from"], f"{name}: doc {doc} chunk {c} score_from")
            ok(m["prompt_sha256"] == sha(j["prompt"].encode()) and m["tokens_sha256"] == sha(ct.tobytes()), f"{name}: doc {doc} chunk {c} sha256")
            ok(T[a] == IMS and T[a + 1] == USER, f"{name}: doc {doc} chunk {c} does not start with <|im_start|>user")
            ok(b == len(T) or (T[b - 2] == IME and T[b - 1] == NL and T[b] == IMS), f"{name}: doc {doc} chunk {c} ends inside a message")
            for o, e in TAGS: ok((T[a:b] == o).sum() == (T[a:b] == e).sum(), f"{name}: doc {doc} chunk {c} unbalanced tool tags")
            sf = H - 1
            ok((gen[go:go + b - a] == G[a - 1:b - 1]).all(), f"{name}: doc {doc} chunk {c} gen differs from the calib golden's")
        ok((toks[to:to + len(ct)] == ct).all(), f"{name}: doc {doc} chunk {c} tokens file")
        if c == "whole": ok((gen[go:go + len(T) - 1] == G).all(), f"{name}: doc {doc} whole gen")
        to += len(ct); go += len(ct) - 1 - sf
    ok(to == len(toks) and go == len(gen), f"{name}: file lengths")
for doc, t in tr.items():
    ch = t["chunks"]; ok(ch[0]["from"] == t["header_tokens"] and ch[-1]["to"] == t["tokens"], f"doc {doc}: chunks do not cover the trace")
    ok(all(x["to"] == y["from"] for x, y in zip(ch, ch[1:])), f"doc {doc}: chunks do not tile")
    ok(t["pool"] == pool[doc], f"doc {doc}: manifest pool")
    ok(("whole" in t) == (pool[doc] == "verify" and "reserved" not in t), f"doc {doc}: whole copy")
    print(f"doc {doc:5d} {t['source']:22} {t['pool']:6} {len(ch)} chunks, tokens {min(c['tokens'] for c in ch)}..{max(c['tokens'] for c in ch)}, "
          f"slice {t['slice']}{' + agent-verify-whole' if 'whole' in t else ''}")
print("\n".join(f"FAIL {f}" for f in fails) or "ALL PASS")
sys.exit(1 if fails else 0)
