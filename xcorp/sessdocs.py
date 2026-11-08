#!/usr/bin/env python3
"""sessdocs.py [--src DIR] [--out DIR] [--cap N] [--max-tool N] [--full] -- stage 1 (render): our pi session archive
as training documents, incrementally (user 2026-11-07: rerun as the archive grows; real token counts). Stage 2 is
xcorp/sessdedup.py (exact + fuzzy dedup, on this stage's output).

Rerun any time: only session files that are new or changed since the last run (size / mtime) are rendered again.
Rendering is pisess.py's: the active branch, Qwen3.8's chat template with tools (reasoning in reasoning_content,
tool calls / responses as the template writes them), long sessions cut at turn boundaries into pieces of <= --cap
tokens; then pisess_redact.py's redaction (secrets, our IPs / key names, e-mails, high-entropy strings) and its
integrity flags. No dedup here, except that attached mining chunks of our own logs (local/mine/chunks) become a
placeholder (they are our sessions again, verbatim).

Kinds (as local/mine/filter.py): main, sub (subagent runs), worker (local/workers), miner (local/mine), idea
(local/papers lenses), other (any other --tmp-- run).

Output (private: never publish; keep under local/ or /data):
  OUT/docs/<kind>-<path>.jsonl  one line per document {source, category, kind, session, file, piece, tokens,
                                assistant_tokens, flags, prompt}; xcorp/tokenize.py OUT/docs works on it
  OUT/stats.json                by kind: files, docs, tokens, assistant_tokens (inside assistant turns: the SFT /
                                distillation targets), turns, redaction hits, flags
  OUT/state.json                per-file size / mtime (the incremental state)
Run: /root/sessvenv/bin/python -I xcorp/sessdocs.py (transformers + jinja2, no torch). Tokenizer: XCORP_TOKENIZER
(default /workspace/models/qwen38-27b-bf16 on the VM). ~2.5 min for the whole archive (1,054 files, 11-07)."""
import argparse, collections, glob, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)        # appended: xcorp/tokenize.py would shadow the stdlib module
os.environ.setdefault("XCORP_TOKENIZER", "/workspace/models/qwen38-27b-bf16")
import pisess, pisess_redact as red

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/workspace/pi-sessions")
ap.add_argument("--out", default=os.path.join(HERE, "..", "local", "sessdocs"))
ap.add_argument("--cap", type=int, default=250000); ap.add_argument("--max-tool", type=int, default=16000)
ap.add_argument("--full", action="store_true", help="re-render everything")
a = ap.parse_args()
OUT = os.path.abspath(a.out); D = f"{OUT}/docs"; os.makedirs(D, exist_ok=True); SP = f"{OUT}/state.json"
st = json.load(open(SP)) if os.path.exists(SP) and not a.full else {"files": {}}
if a.full:
    for f in glob.glob(f"{D}/*.jsonl"): os.remove(f)

from transformers import AutoTokenizer
from transformers.utils import logging as hflog
hflog.set_verbosity_error()                 # "longer than the model's maximum" warnings: we only count
tok = AutoTokenizer.from_pretrained(os.environ["XCORP_TOKENIZER"])
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
render = lambda ms, tl: tok.apply_chat_template(ms, tools=tl, tokenize=False)


class NoDedup:
    """pisess.dedupe_files' seen-set, disabled: dedup is stage 2's job"""
    def __contains__(self, h): return False
    def add(self, h): pass


def kind_of(rel, head):
    if rel.startswith("--workspace--/"): return "main" if rel.count("/") == 1 else "sub"
    if "local/mine/brief.md" in head or "local/mine/chunks" in head: return "miner"
    if "papers/run/instr-" in head: return "idea"
    if "local/workers/instr3/" in head: return "worker"
    return "other"


def convert(path, rel, kind):
    E = []
    for l in open(path, errors="replace"):
        try: E.append(json.loads(l))
        except Exception: pass
    pisess.SEEN_FILES = NoDedup()
    msgs, tl = pisess.convert(pisess.active_path(E), a.max_tool)
    c = collections.Counter()
    if not any(m["role"] == "assistant" for m in msgs): return [], c
    sid = rel.split("/")[-3] if rel.endswith("/session.jsonl") else os.path.basename(rel)[:-6]
    base = n_tok(render([{"role": "user", "content": "x"}], tl)) - 1
    pieces, cur, cur_n = [], [], base
    for tu in pisess.turns(msgs):
        c["turns"] += 1
        try: n = n_tok(render(tu, tl)) - base
        except Exception: c["render_fail"] += 1; continue
        if n > a.cap - base: c["turns_over_cap"] += 1; continue
        if cur and cur_n + n > a.cap: pieces.append(cur); cur, cur_n = [], base
        cur += tu; cur_n += n
    if cur: pieces.append(cur)
    docs, hits = [], collections.Counter()
    for i, p in enumerate(pieces):
        if p[0]["role"] != "user": p = [{"role": "user", "content": "(continued session)"}] + p
        try: prompt = red.redact(render(p, tl), hits)
        except Exception: c["render_fail"] += 1; continue
        if "<|im_start|>assistant" not in prompt: continue
        at = sum(n_tok(s) for s in prompt.split("<|im_start|>")[1:] if s.startswith("assistant"))
        docs.append({"source": "pi-sessions", "category": "agentic", "kind": kind, "session": sid, "file": rel,
                     "piece": i, "tokens": n_tok(prompt), "assistant_tokens": at, "flags": red.flags_of(prompt),
                     "prompt": prompt})
    c.update({f"redact_{k}": v for k, v in hits.items()})
    return docs, c


t0 = time.time()
files = sorted(glob.glob(f"{a.src}/--workspace--/*.jsonl") + glob.glob(f"{a.src}/--workspace--/*/*/run-0/session.jsonl")
               + glob.glob(f"{a.src}/--tmp--/*.jsonl"))
seen, n_new = set(), 0
for path in files:
    rel = os.path.relpath(path, a.src); seen.add(rel); s = os.stat(path)
    old = st["files"].get(rel)
    if old and old["size"] == s.st_size and old["mtime"] == s.st_mtime: continue
    with open(path, errors="replace") as fh: head = fh.read(400000)
    kind = kind_of(rel, head)
    docs, c = convert(path, rel, kind)
    name = f"{kind}-{re.sub(r'[^A-Za-z0-9_.-]', '_', rel[:-6].replace('/', '__'))[:180]}.jsonl"
    if old and old.get("out") and old["out"] != name and os.path.exists(f"{D}/{old['out']}"): os.remove(f"{D}/{old['out']}")
    if docs:
        with open(f"{D}/{name}.part", "w") as fo:
            for d in docs: fo.write(json.dumps(d, ensure_ascii=False) + "\n")
        os.replace(f"{D}/{name}.part", f"{D}/{name}")
    elif os.path.exists(f"{D}/{name}"): os.remove(f"{D}/{name}")
    st["files"][rel] = {"size": s.st_size, "mtime": s.st_mtime, "kind": kind, "out": name if docs else None,
                        "docs": len(docs), "tokens": sum(d["tokens"] for d in docs),
                        "assistant_tokens": sum(d["assistant_tokens"] for d in docs),
                        "flags": dict(collections.Counter(f for d in docs for f in d["flags"])), "counts": dict(c)}
    n_new += 1
    if n_new % 100 == 0: print(f"{n_new} files rendered ({time.time() - t0:.0f} s)", flush=True)
for rel in [r for r in st["files"] if r not in seen]:     # gone from the archive (it is additive: rare)
    o = st["files"].pop(rel).get("out")
    if o and os.path.exists(f"{D}/{o}"): os.remove(f"{D}/{o}")

tot = collections.defaultdict(collections.Counter)
for v in st["files"].values():
    T = tot[v["kind"]]; T["files"] += 1; T["docs"] += v["docs"]; T["tokens"] += v["tokens"]
    T["assistant_tokens"] += v["assistant_tokens"]; T.update(v.get("counts", {}))
    T.update({f"flag_{k}": n for k, n in v.get("flags", {}).items()})
allc = sum(tot.values(), collections.Counter())
json.dump(st, open(SP + ".part", "w")); os.replace(SP + ".part", SP)
json.dump({"updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "files_rendered_this_run": n_new,
           "by_kind": {k: dict(v) for k, v in sorted(tot.items())}, "all": dict(allc)},
          open(f"{OUT}/stats.json", "w"), indent=1)
print(f"{n_new} files rendered in {time.time() - t0:.0f} s")
for k, v in sorted(tot.items()):
    print(f"{k:7s} files {v['files']:4d} docs {v['docs']:4d} tokens {v['tokens']/1e6:7.2f}M  assistant {v['assistant_tokens']/1e6:6.2f}M")
print(f"ALL     tokens {allc['tokens']/1e6:.2f}M  assistant {allc['assistant_tokens']/1e6:.2f}M")
