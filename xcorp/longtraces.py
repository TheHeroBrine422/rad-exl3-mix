#!/usr/bin/env python3
"""longtraces.py RAW_DIR OUT_JSONL [--min 300000] [--cap 1000000] [--max-tool 16000] -- long agentic sessions from
public trace datasets (Claude Code and pi session logs, as uploaded to the Hugging Face Hub), one document per
session, rendered with DeepSeek-V4.1's encoding (xcorp/build.py's ds_render) and tokenized with its tokenizer.

RAW_DIR/<owner>__<repo>/**/*.jsonl, one session per file. Formats:
  * Claude Code: entries {type: user|assistant, message, uuid, parentUuid, isSidechain}; the main chain (no
    sidechains / subagents), assistant entries of one message.id merged; thinking -> reasoning_content, tool_use ->
    tool_calls, tool_result blocks -> tool messages; compact summaries, meta / command entries dropped.
  * pi (session v3): xcorp/pisess.py's converter (active branch, toolCall / toolResult).
Tool schemas are not logged by either: each session gets the tool names it used with the argument keys it passed.
Tool results over --max-tool characters keep head + tail. Secrets scrubbed (pisess.SECRETS). A turn already emitted
by another session (forks / continuations) is dropped. Sessions under --min tokens are skipped; longer than --cap are
cut at the last turn boundary that fits. Output lines {source, category: "long-agentic", kind: "chat", tokens,
prompt, session, model, license}.
Env: XCORP_TOKENIZER (dir with tokenizer.json), XCORP_DS_ENCODING (encoding.py)."""
import argparse, glob, hashlib, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pisess                                   # noqa: E402  (converter, scrub, clip, turns)
from build import DSTok, ds_render              # noqa: E402


def cc_text(c):
    if isinstance(c, str): return c
    out = []
    for p in c or []:
        if not isinstance(p, dict): continue
        if p.get("type") == "text": out.append(p.get("text", ""))
        elif p.get("type") == "image": out.append("[image]")
    return "\n".join(out)


def claude_code(E, max_tool):
    """Main-chain messages of a Claude Code session log."""
    E = [e for e in E if e.get("type") in ("user", "assistant") and not e.get("isSidechain")]
    by = {e.get("uuid"): e for e in E}
    last = E[-1] if E else None
    chain, cur, seen = [], last, set()
    while cur is not None and cur.get("uuid") not in seen:      # active branch back from the last entry
        seen.add(cur.get("uuid")); chain.append(cur); cur = by.get(cur.get("parentUuid"))
    chain.reverse()
    if len(chain) < len(E) // 2: chain = E                       # broken parent links: file order
    msgs, tools, models = [], {}, {}
    for e in chain:
        m = e.get("message") or {}
        if e["type"] == "user":
            if e.get("isCompactSummary") or e.get("isMeta"): continue
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "tool_result" for p in c):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "tool_result":
                        msgs.append({"role": "tool", "tool_call_id": p.get("tool_use_id", ""),
                                     "content": pisess.scrub(pisess.clip(cc_text(p.get("content")), max_tool))})
                continue
            s = cc_text(c)
            if s.strip().startswith(("<command-name>", "<local-command-stdout>", "<command-message>")): continue
            if s.strip(): msgs.append({"role": "user", "content": pisess.scrub(s)})
        else:
            if m.get("model") and not str(m["model"]).startswith("<"): models[m["model"]] = models.get(m["model"], 0) + 1
            think, text, calls = [], [], []
            for p in m.get("content") or []:
                if not isinstance(p, dict): continue
                if p.get("type") == "thinking" and p.get("thinking"): think.append(p["thinking"])
                elif p.get("type") == "text" and p.get("text"): text.append(p["text"])
                elif p.get("type") == "tool_use":
                    args = p.get("input") or {}
                    if not isinstance(args, dict): args = {"input": args}
                    args = json.loads(pisess.scrub(json.dumps(args)))
                    calls.append({"id": p.get("id", ""), "type": "function",
                                  "function": {"name": p.get("name", "tool"), "arguments": args}})
                    tools.setdefault(p.get("name", "tool"), set()).update(args.keys())
            if not (think or text or calls): continue
            prev = msgs[-1] if msgs else None
            mid = m.get("id")
            if prev is not None and prev["role"] == "assistant" and mid and prev.get("_mid") == mid:
                if think: prev["reasoning_content"] = (prev.get("reasoning_content", "") + "\n\n" + "\n\n".join(think)).strip()
                if text: prev["content"] = (prev["content"] + "\n\n" + "\n\n".join(text)).strip()
                if calls: prev.setdefault("tool_calls", []).extend(calls)
                continue
            a = {"role": "assistant", "content": pisess.scrub("\n\n".join(text)), "_mid": mid}
            if think: a["reasoning_content"] = pisess.scrub("\n\n".join(think))
            if calls: a["tool_calls"] = calls
            msgs.append(a)
    for a in msgs: a.pop("_mid", None)
    model = max(models, key=models.get) if models else None
    tl = [{"type": "function", "function": {"name": n, "description": f"tool {n}",
           "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in sorted(ks)}}}}
          for n, ks in sorted(tools.items())]
    return msgs, tl, model


def pi_session(E, max_tool):
    model = next((e.get("modelId") for e in E if e.get("type") == "model_change"), None)
    msgs, tl = pisess.convert(pisess.active_path(E), max_tool)
    return msgs, tl, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--min", type=int, default=300000); ap.add_argument("--cap", type=int, default=1000000)
    ap.add_argument("--max-tool", type=int, default=16000)
    ap.add_argument("--licenses", help="json {repo_dir: license}")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("ds_encoding", os.environ["XCORP_DS_ENCODING"])
    enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
    tok = DSTok(os.path.join(os.environ["XCORP_TOKENIZER"], "tokenizer.json"))
    lic = json.load(open(a.licenses)) if a.licenses else {}
    seen_turns, stats, fo = set(), [], open(a.out + ".part", "w")
    files = sorted(glob.glob(os.path.join(a.raw, "*", "**", "*.jsonl"), recursive=True), key=os.path.getsize, reverse=True)
    for f in files:
        repo = os.path.relpath(f, a.raw).split(os.sep)[0]
        try:
            E = [json.loads(l) for l in open(f, encoding="utf-8", errors="replace") if l.strip().startswith("{")]
        except Exception:
            stats.append((repo, f, "parse", 0)); continue
        if not E: continue
        if any(e.get("type") == "session" for e in E[:3]): msgs, tools, model = pi_session(E, a.max_tool)
        elif any(e.get("type") in ("user", "assistant") for e in E): msgs, tools, model = claude_code(E, a.max_tool)
        else:
            stats.append((repo, f, "format", 0)); continue
        keep = []
        for t in pisess.turns(msgs):
            h = hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
            if h in seen_turns: continue
            seen_turns.add(h); keep.append(t)
        if not keep or not any(m["role"] == "assistant" for t in keep for m in t):
            stats.append((repo, f, "empty", 0)); continue
        flat = [m for t in keep for m in t]
        if flat[0]["role"] != "user": flat = [{"role": "user", "content": "(session continues)"}] + flat
        if "<\uff5c" in json.dumps(flat, ensure_ascii=False):
            stats.append((repo, f, "special-token", 0)); continue
        try: prompt = ds_render(enc, flat, tools or None)
        except Exception as ex:
            stats.append((repo, f, f"render:{type(ex).__name__}", 0)); continue
        n = len(tok(prompt)["input_ids"])
        if n > a.cap:                                            # cut at an assistant message from the end
            units = []                                           # [user / tool msgs...] + each assistant msg
            for m in flat:                                       # with the tool results that follow it
                if m["role"] == "assistant" or not units: units.append([m])
                else: units[-1].append(m)
            def build(k):
                fl = [m for u in units[:k] for m in u]
                return fl if fl[0]["role"] == "user" else [{"role": "user", "content": "(session continues)"}] + fl
            lo, hi = 1, len(units)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(tok(ds_render(enc, build(mid), tools or None))["input_ids"]) <= a.cap: lo = mid
                else: hi = mid - 1
            prompt = ds_render(enc, build(lo), tools or None); n = len(tok(prompt)["input_ids"])
            if n > a.cap:
                stats.append((repo, f, "too-long-unit", n)); continue
        stats.append((repo, f, "ok" if n >= a.min else "short", n))
        if n < a.min: continue
        fo.write(json.dumps({"source": repo.replace("__", "/", 1), "category": "long-agentic", "kind": "chat",
                             "tokens": n, "prompt": prompt, "session": os.path.basename(f), "model": model,
                             "license": lic.get(repo)}, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"{repo} {os.path.basename(f)} {model} {n}", flush=True)
    fo.close(); os.replace(a.out + ".part", a.out)
    json.dump(stats, open(a.out + ".stats.json", "w"))
    ok = [s for s in stats if s[2] == "ok"]
    print(f"DONE {len(ok)} sessions >= {a.min}, {sum(s[3] for s in ok)} tokens; "
          f"{len(stats)} files", flush=True)


if __name__ == "__main__":
    main()
