#!/usr/bin/env python3
"""chunk.py CORPUS POOLS OUTDIR [--tokenizer DIR] [--max N] [--prefer-task F] [--gold TRACEDIR] [--reserved]
-- the 8 long agentic calib traces cut into turn-aligned chunks of <= N tokens (default 32768), each chunk = the trace's
system message (system prompt + tool definitions: the unscored header) + consecutive messages starting at a user-role
message (a task or a tool result: Qwen's template renders tool results as user messages), as xcorp/kldcorpus.py --trace
cuts our own sessions. A chunk ends before a user message, so no message, tool call or tool result is ever split.
Cuts: greedy over whole turns with a body cap; when the next turn doesn't fit and a task statement starts in the
chunk's last (1 - F) share of the cap (default F = 0.75), the chunk is cut there instead, so the next chunk starts with
its task. The cap is the smallest that gives as few chunks as the full room (N - header) does (bisection), so a trace's
chunks come out about equal instead of full ones plus a short tail.

Scoring masks the header: corpus lines carry score_from = header - 1, so the scored rows predict exactly the chunk's
body tokens (rows = body tokens). gen (the official score's rows: tokens the model writes itself) follows the calib
golden's rule, checked against it with --gold: the tokens of an assistant message after `<|im_start|>assistant\\n`,
through its `<|im_end|>`.

Pools come from POOLS (pools-v3.json's per-document "pool", never its stale "traces" field); a chunk stays in its
trace's pool. Outputs in OUTDIR (deterministic: same inputs, byte-identical files; "32" = round(N / 1024)):
  agent-search32.jsonl    search-pool chunks (golden: agent-search32)
  agent-verify32.jsonl    verify-pool chunks except the reserved traces (golden: agent-verify32)
  qad32.jsonl             qad-pool chunks (QAD / healing data; no golden)
  agent-verify-whole.jsonl  the routine verify traces whole (the original calib lines): read-only full-length copies
  verify-reserved32.jsonl only with --reserved: chunks of the traces held for V3 monitor / final (RESERVED)
  NAME.gen.u8, NAME.tokens.i32  per chunk corpus: the gen mask of its scored rows and its tokens, in corpus order
                          (what the golden's gen.u8 / tokens.i32 must be; agent32.sh's finalize writes / checks them)
  manifest.json           per trace and chunk: pool, token ranges, header length, rows, sha256s
--gold TRACEDIR: t<doc>.i32 / g<doc>.u8 = the trace's tokens / gen rows cut from the calib golden (pulltraces.py):
tokenisation and the gen rule must reproduce them exactly. Docs: eval/traces/README.md."""
import argparse, hashlib, json, os, sys
import numpy as np
from tokenizers import Tokenizer

# Verify-pool traces already reserved for the V3 line's long-context monitor / final (eval/pools/ledger.jsonl): chunked
# into the manifest (deterministic data prep), but no corpus unless --reserved and no golden.
RESERVED = {2300: "golden-ver3-long (V3 monitor, from 8192)", 4723: "golden-ver4-long (V3 final, from 16384)"}

ap = argparse.ArgumentParser()
ap.add_argument("corpus"); ap.add_argument("pools"); ap.add_argument("out")
ap.add_argument("--tokenizer", default="/workspace/models/qwen38-27b-bf16/tokenizer.json")
ap.add_argument("--max", type=int, default=32768, help="tokens per chunk, header included")
ap.add_argument("--prefer-task", type=float, default=0.75)
ap.add_argument("--gold", default=None)
ap.add_argument("--reserved", action="store_true")
a = ap.parse_args()
X = round(a.max / 1024)                           # slice names carry the chunk size in k tokens: agent-search32, ...
SLICE = {"search": f"agent-search{X}", "verify": f"agent-verify{X}", "qad": f"qad{X}"}


def sha(b): return hashlib.sha256(b).hexdigest()


tk = Tokenizer.from_file(a.tokenizer)
IMS, IME = tk.token_to_id("<|im_start|>"), tk.token_to_id("<|im_end|>")
ASST, NL = tk.token_to_id("assistant"), tk.token_to_id("Ċ")
enc = lambda s: np.array(tk.encode(s, add_special_tokens=False).ids, np.int32)
pool = json.load(open(a.pools))["pool"]
raw = open(a.corpus, "rb").read()
lines = raw.decode().rstrip("\n").split("\n")     # not splitlines(): U+2028 etc. inside a JSON string are not line ends
assert len(lines) == len(pool), (len(lines), len(pool))
traces = [i for i, l in enumerate(lines) if json.loads(l).get("source", "").startswith("agent-trace/")]
os.makedirs(a.out, exist_ok=True)
man = {"tool": "eval/traces/chunk.py", "max_tokens": a.max, "prefer_task": a.prefer_task,
       "corpus": os.path.basename(a.corpus), "corpus_sha256": sha(raw), "pools": os.path.basename(a.pools),
       "pools_sha256": sha(open(a.pools, "rb").read()), "tokenizer_sha256": sha(open(a.tokenizer, "rb").read()),
       "gold_checked": a.gold is not None, "traces": [], "slices": {}}
out = {}            # slice name -> list of (corpus line, tokens, gen)


def put(name, line, toks, gen, member): out.setdefault(name, []).append((line, toks, gen, member))


for i in traces:
    j = json.loads(lines[i]); p = j["prompt"]; t = enc(p)
    if a.gold:
        g = np.fromfile(f"{a.gold}/t{i}.i32", np.int32)
        assert len(g) == len(t) and (g == t).all(), f"doc {i}: tokens differ from the calib golden's"
    st = np.flatnonzero(t == IMS)
    cs = []; k = p.find("<|im_start|>")
    while k >= 0: cs.append(k); k = p.find("<|im_start|>", k + 1)
    assert len(cs) == len(st), f"doc {i}: {len(cs)} '<|im_start|>' strings, {len(st)} tokens"
    ends = np.append(st[1:], len(t)); cends = cs[1:] + [len(p)]
    kind = []
    for s, e in zip(st, ends):
        r = tk.decode([int(x) for x in t[s + 1:s + 2]], skip_special_tokens=False)
        if r == "assistant": kind.append("A")
        elif r == "system": kind.append("S")
        elif r == "user": kind.append("R" if "<tool_response>" in tk.decode([int(x) for x in t[s + 2:s + 6]], skip_special_tokens=False) else "U")
        else: raise SystemExit(f"doc {i}: message at {s} has role {r!r}")
    assert kind[0] == "S" and "S" not in kind[1:], f"doc {i}: not one leading system message"
    # gen rule, per target token
    gen = np.zeros(len(t), np.uint8)
    for s, e, c in zip(st, ends, kind):
        if c != "A": continue
        assert t[s + 1] == ASST and t[s + 2] == NL, f"doc {i}: assistant header at {s}"
        ie = np.flatnonzero(t[s:e] == IME); assert len(ie), f"doc {i}: assistant message at {s} has no <|im_end|>"
        gen[s + 3:s + ie[0] + 1] = 1
    if a.gold:
        g = np.fromfile(f"{a.gold}/g{i}.u8", np.uint8)
        assert len(g) == len(t) - 1 and (g == gen[1:]).all(), f"doc {i}: gen rule differs from the calib golden's gen.u8 at {int((g != gen[1:]).sum())} rows"
    H = int(st[1]); hdr = p[:cs[1]]; room = a.max - H
    assert room >= 1024, f"doc {i}: header {H} leaves no room"
    users = [k for k in range(1, len(st)) if kind[k] in "UR"]

    def cut(cap):
        """greedy chunks of <= cap body tokens, whole turns, a task start preferred late in a chunk; None if a turn
        doesn't fit"""
        chunks, u = [], 0
        while u < len(users):
            a0 = int(st[users[u]]); best = None; v = u + 1
            while v <= len(users):                    # extend by whole turns (up to the next user message)
                end = int(st[users[v]]) if v < len(users) else len(t)
                if end - a0 > cap: break
                best = v; v += 1
            if best is None: return None
            if best < len(users):                     # not the trace's end: prefer a task start late in the chunk
                for w in range(best - 1, u, -1):
                    if kind[users[w]] == "U":
                        if int(st[users[w]]) - a0 >= a.prefer_task * cap: best = w
                        break
            chunks.append((u, a0, int(st[users[best]]) if best < len(users) else len(t))); u = best
        return chunks

    g0 = cut(room)
    if g0 is None: raise SystemExit(f"doc {i}: a turn does not fit {room} tokens after the header: raise --max")
    lo, hi = 1, room                                  # balance: the smallest cap that still gives greedy's count
    while lo < hi:
        mid = (lo + hi) // 2; c = cut(mid)
        if c is not None and len(c) <= len(g0): hi = mid
        else: lo = mid + 1
    chunks = cut(hi); cap = hi
    assert chunks is not None and len(chunks) <= len(g0)
    tr = {"doc": i, "source": j["source"], "pool": pool[i], "tokens": int(len(t)), "tokens_sha256": sha(t.tobytes()),
          "header_tokens": H, "header_sha256": sha(hdr.encode()), "messages": len(st), "tasks": kind.count("U"),
          "gen_rows": int(gen[1:].sum()), "body_cap": cap}
    if i in RESERVED: tr["reserved"] = RESERVED[i]; name = f"verify-reserved{X}" if a.reserved else None
    else: name = SLICE[pool[i]]
    if pool[i] == "verify" and i not in RESERVED:
        tr["whole"] = {"slice": "agent-verify-whole", "line_sha256": sha(lines[i].encode())}
        put("agent-verify-whole", lines[i], t, gen[1:], [i, "whole"])
    tr["slice"] = name; tr["chunks"] = []
    body_cover = 0
    for c, (u0, a0, b) in enumerate(chunks):
        ca = cs[int(np.searchsorted(st, a0))]; cb = cends[int(np.searchsorted(st, b)) - 1]
        prompt = hdr + p[ca:cb]; ct = enc(prompt)
        want = np.concatenate([t[:H], t[a0:b]])
        assert len(ct) == len(want) and (ct == want).all(), f"doc {i} chunk {c}: re-tokenised chunk differs from the trace's tokens"
        assert len(ct) <= a.max
        sf = H - 1; cg = gen[a0:b]                     # rows sf .. n-2 predict tokens H .. n-1 = the body
        line = json.dumps({"source": j["source"], "category": j.get("category", ""), "kind": j.get("kind", ""),
                           "tokens": int(len(ct)), "orig": i, "chunk": c, "from": a0, "to": b, "score_from": sf,
                           "prompt": prompt}, ensure_ascii=False)
        tr["chunks"].append({"chunk": c, "from": a0, "to": b, "first": kind[users[u0]], "tokens": int(len(ct)),
                             "header_tokens": H, "score_from": sf, "rows": int(b - a0), "gen_rows": int(cg.sum()),
                             "prompt_sha256": sha(prompt.encode()), "tokens_sha256": sha(ct.tobytes())})
        body_cover += b - a0
        if name: put(name, line, ct, cg, [i, c])
    assert chunks[0][1] == H and chunks[-1][2] == len(t) and all(x[2] == y[1] for x, y in zip(chunks, chunks[1:]))
    assert body_cover == len(t) - H
    man["traces"].append(tr)
    print(f"{i:5d} {j['source']:22} {pool[i]:6} {len(t):7d} tok, header {H}, {len(chunks)} chunks "
          f"({min(c['tokens'] for c in tr['chunks'])}..{max(c['tokens'] for c in tr['chunks'])} tok, "
          f"{sum(c['first'] == 'U' for c in tr['chunks'])} start at a task) -> {name or '-'}"
          + (f" [reserved: {RESERVED[i]}]" if i in RESERVED else ""), file=sys.stderr)

for name, items in sorted(out.items()):
    body = "".join(l + "\n" for l, _, _, _ in items).encode()
    toks = np.concatenate([x for _, x, _, _ in items]); gen = np.concatenate([g for _, _, g, _ in items])
    for fn, b in ((f"{name}.jsonl", body), (f"{name}.tokens.i32", toks.tobytes()), (f"{name}.gen.u8", gen.tobytes())):
        with open(f"{a.out}/{fn}.part", "wb") as f: f.write(b)
        os.replace(f"{a.out}/{fn}.part", f"{a.out}/{fn}")
    docs = [json.loads(l) for l, _, _, _ in items]
    man["slices"][name] = {"file": f"{name}.jsonl", "sha256": sha(body), "docs": len(items), "tokens": int(len(toks)),
                           "rows": int(len(gen)), "gen_rows": int(gen.sum()), "max_doc_tokens": max(d["tokens"] for d in docs),
                           "members": [m for _, _, _, m in items]}
    print(f"{name}: {len(items)} docs, {len(toks)} tokens, {len(gen)} rows, {int(gen.sum())} gen rows", file=sys.stderr)
with open(f"{a.out}/manifest.json.part", "w") as f: json.dump(man, f, indent=1); f.write("\n")
os.replace(f"{a.out}/manifest.json.part", f"{a.out}/manifest.json")
print(f"CHUNK DONE {a.out}/manifest.json {sha(open(f'{a.out}/manifest.json', 'rb').read())}", file=sys.stderr)
