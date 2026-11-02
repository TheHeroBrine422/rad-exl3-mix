#!/usr/bin/env python3
"""pisess.py SESSIONS_DIR OUTDIR [--cap 250000] [--max-tool 16000] -- our own pi session logs as agentic traces in the
challenge's rendering (an allowed training / statistics corpus, user 2026-11-02; never published: private content).

Per session file (`SESSIONS_DIR/**/*.jsonl`: main sessions and subagent runs, or the print-mode runs in --tmp--,
whose one user message is attached files: each file's content is kept the first time it is seen, repeats and
mining chunks of our own logs become a placeholder): the active branch (parentId chain back from the last entry), rendered with
Qwen3.8's chat template like xcorp/build.py (tools in the system block, reasoning in reasoning_content):
  user text -> user; assistant thinking -> reasoning_content, text -> content, toolCall -> tool_calls;
  toolResult -> tool (images -> "[image]", text over --max-tool chars keeps head + tail); bashExecution (commands the
  user ran) and custom_message (harness notices) -> user; compaction summaries, web-search caches, model changes and
  empty / aborted assistant messages are dropped.
Tools: the names the session used, each with the argument keys it was called with (pi does not log schemas).
Secrets scrubbed (HF / API tokens, private-key blocks). A turn (a user message and everything up to the next one)
already emitted from another file is skipped (continued / forked sessions). Long sessions are cut at turn
boundaries into pieces of <= --cap tokens, each starting with the system block.
Output OUTDIR/pisess.jsonl lines {source, category, kind, tokens, prompt, session, piece} (xcorp/tokenize.py turns it
into meta/tokens.i32 + kld.json) and OUTDIR/pisess.stats.json."""
import argparse, glob, hashlib, json, os, re

TOKENIZER = os.environ.get("XCORP_TOKENIZER", "/data/models/qwen38-27b-bf16")
SECRETS = [(re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "hf_REDACTED"),
           (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}\b"), "sk-REDACTED"),
           (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "ghp_REDACTED"),
           (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[PRIVATE KEY REDACTED]")]


def scrub(s):
    for p, r in SECRETS: s = p.sub(r, s)
    return s


def clip(s, n):
    return s if len(s) <= n else s[: n // 2] + f"\n[... {len(s) - n} characters cut ...]\n" + s[-n // 2:]


def texts(c):
    if isinstance(c, str): return c
    return "\n".join(p.get("text", "") if p.get("type") == "text" else "[image]" for p in c or [] if isinstance(p, dict))


FILEBLOCK = re.compile(r'<file name="([^"]*)">\n?(.*?)</file>', re.S)
SEEN_FILES = set()


def dedupe_files(s):
    """print-mode runs attach the same files again and again: keep an attached file's content the first time it is
    seen (any session), replace repeats -- and mining chunks of our own session logs, which duplicate the main
    sessions -- by a placeholder"""
    def rep(m):
        name, body = m.group(1), m.group(2)
        h = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()
        if "/local/mine/chunks" in name or h in SEEN_FILES:
            return f'<file name="{name}">[contents as attached earlier]</file>'
        SEEN_FILES.add(h); return m.group(0)
    return FILEBLOCK.sub(rep, s)


def active_path(E):
    by = {e["id"]: e for e in E if "id" in e}
    last = next((e for e in reversed(E) if e.get("type") == "message"), None)
    path, cur = [], last
    while cur is not None:
        path.append(cur); cur = by.get(cur.get("parentId"))
    return list(reversed(path))


def convert(path, max_tool):
    msgs, tools = [], {}
    for e in path:
        t = e.get("type")
        if t == "custom_message":
            msgs.append({"role": "user", "content": scrub(str(e.get("content", "")))}); continue
        if t != "message": continue
        m = e["message"]; r = m.get("role")
        if r == "user":
            s = dedupe_files(texts(m.get("content")))
            if s.strip(): msgs.append({"role": "user", "content": scrub(s)})
        elif r == "bashExecution":
            msgs.append({"role": "user", "content": scrub(f"$ {m.get('command', '')}\n{clip(m.get('output', ''), max_tool)}")})
        elif r == "assistant":
            think, text, calls = [], [], []
            for p in m.get("content") or []:
                if p.get("type") == "thinking" and p.get("thinking"): think.append(p["thinking"])
                elif p.get("type") == "text" and p.get("text"): text.append(p["text"])
                elif p.get("type") == "toolCall":
                    args = p.get("arguments") or {}
                    if not isinstance(args, dict): args = {"input": args}
                    args = json.loads(scrub(json.dumps(args)))
                    calls.append({"type": "function", "function": {"name": p.get("name", "tool"), "arguments": args}})
                    tools.setdefault(p.get("name", "tool"), set()).update(args.keys())
            if not (think or text or calls): continue       # aborted / empty
            a = {"role": "assistant", "content": scrub("\n\n".join(text))}
            if think: a["reasoning_content"] = scrub("\n\n".join(think))
            if calls: a["tool_calls"] = calls
            msgs.append(a)
        elif r == "toolResult":
            msgs.append({"role": "tool", "content": scrub(clip(texts(m.get("content")), max_tool))})
    tl = [{"type": "function", "function": {"name": n, "description": f"pi tool {n}",
           "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in sorted(ks)}}}} for n, ks in sorted(tools.items())]
    return msgs, tl


def turns(msgs):
    out, cur = [], []
    for m in msgs:
        if m["role"] == "user" and cur and any(x["role"] != "user" for x in cur): out.append(cur); cur = []
        cur.append(m)
    if cur: out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions"); ap.add_argument("out")
    ap.add_argument("--cap", type=int, default=250000); ap.add_argument("--max-tool", type=int, default=16000)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    render = lambda ms, tl: tok.apply_chat_template(ms, tools=tl, tokenize=False)
    os.makedirs(a.out, exist_ok=True)
    fo = open(f"{a.out}/pisess.jsonl.part", "w"); seen = set()
    st = {"files": 0, "pieces": 0, "tokens": 0, "turns": 0, "turns_dup": 0, "turns_cut": 0, "render_fail": 0}
    for f in sorted(glob.glob(f"{a.sessions}/**/*.jsonl", recursive=True)):
        E = []
        for l in open(f, errors="replace"):
            try: E.append(json.loads(l))
            except Exception: pass
        msgs, tl = convert(active_path(E), a.max_tool)
        if not any(m["role"] == "assistant" for m in msgs): continue
        st["files"] += 1; sid = os.path.basename(f)[:-6]
        base = n_tok(render([{"role": "user", "content": "x"}], tl)) - 1
        pieces, cur, cur_n = [], [], base
        for tu in turns(msgs):
            h = hashlib.sha256(json.dumps(tu, sort_keys=True).encode()).hexdigest()
            if h in seen: st["turns_dup"] += 1; continue
            seen.add(h); st["turns"] += 1
            try: n = n_tok(render(tu, tl)) - base
            except Exception: st["render_fail"] += 1; continue
            if n > a.cap - base: st["turns_cut"] += 1; continue  # a single turn over the cap: dropped
            if cur and cur_n + n > a.cap: pieces.append(cur); cur, cur_n = [], base
            cur += tu; cur_n += n
        if cur: pieces.append(cur)
        for i, p in enumerate(pieces):
            if p[0]["role"] != "user": p = [{"role": "user", "content": "(continued session)"}] + p
            try: prompt = render(p, tl)
            except Exception: st["render_fail"] += 1; continue
            k = n_tok(prompt)
            fo.write(json.dumps({"source": "pi-sessions", "category": "agentic", "kind": "trace", "tokens": k,
                                 "prompt": prompt, "session": sid, "piece": i}, ensure_ascii=False) + "\n")
            st["pieces"] += 1; st["tokens"] += k
        print(f"{sid[:40]}: {len(pieces)} pieces, total {st['tokens']} tokens", flush=True)
    fo.close(); os.replace(f"{a.out}/pisess.jsonl.part", f"{a.out}/pisess.jsonl")
    json.dump(st, open(f"{a.out}/pisess.stats.json", "w"), indent=1); print("DONE", json.dumps(st))


if __name__ == "__main__":
    main()
