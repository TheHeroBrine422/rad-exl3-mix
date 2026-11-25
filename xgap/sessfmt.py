"""sessfmt.py -- public agent-trace logs (Claude Code JSONL, pi session v3 JSONL) -> structured messages for the Qwen3.8
chat template. Converters derived from xcorp/longtraces.py (Claude Code) and the pi converter of xcorp/pisess.py, trimmed
to the public-dataset case (no private-session helpers) and with typed tool schemas inferred from the calls seen.

msgs: user / assistant (content, reasoning_content, tool_calls) / tool. Tool schemas are not logged by either format:
each session gets the tools it used, parameters typed from the values passed, `required` = keys present in every call."""
import json, os, re

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REAL = {}       # harness format -> {tool name: {name, description, parameters}}: real definitions of public harnesses
for _fmt, _f in (("claude-code", "claude-code-tools.json"), ("pi", "pi-tools.json")):
    REAL[_fmt] = json.load(open(os.path.join(_DATA, _f)))["tools"]

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


def cc_text(c):
    if isinstance(c, str): return c
    out = []
    for p in c or []:
        if not isinstance(p, dict): continue
        if p.get("type") == "text": out.append(p.get("text", ""))
        elif p.get("type") == "image": out.append("[image]")
    return "\n".join(out)


class Tools:
    """name -> {key -> set(json types)}, plus the number of calls and per-key counts, for the inferred schema."""
    def __init__(self): self.t = {}

    def see(self, name, args):
        d = self.t.setdefault(name, {"n": 0, "keys": {}})
        d["n"] += 1
        for k, v in args.items():
            d["keys"].setdefault(k, {"n": 0, "types": set()})
            d["keys"][k]["n"] += 1
            d["keys"][k]["types"].add(_jt(v))

    def schema(self, fmt):
        """-> (tool definitions, {"real": [...], "inferred": [...]}). A tool gets its harness's real definition (extract_cc_tools.py /
        extract_pi_tools.mjs: names, parameter schemas, descriptions) when the harness is known and every argument key seen in the
        session is one of that definition's parameters; otherwise a definition inferred from the calls."""
        out, info = [], {"real": [], "inferred": []}
        for name, d in sorted(self.t.items()):
            r = REAL.get(fmt, {}).get(name)
            if r and set(d["keys"]) <= set(r["parameters"].get("properties", {})):
                out.append({"type": "function", "function": {"name": name, "description": r["description"], "parameters": r["parameters"]}})
                info["real"].append(name); continue
            info["inferred"].append(name)
            props, req = {}, []
            for k, v in sorted(d["keys"].items()):
                props[k] = {"type": next(iter(v["types"]))} if len(v["types"]) == 1 else {}
                if v["n"] == d["n"]: req.append(k)
            fn = {"name": name, "description": f"Tool {name}.", "parameters": {"type": "object", "properties": props}}
            if req: fn["parameters"]["required"] = req
            out.append({"type": "function", "function": fn})
        return out, info


def _jt(v):
    if isinstance(v, bool): return "boolean"
    if isinstance(v, int): return "integer"
    if isinstance(v, float): return "number"
    if isinstance(v, list): return "array"
    if isinstance(v, dict): return "object"
    return "string"


def claude_code(E, max_tool):
    """Main-chain messages of a Claude Code session log: active branch, no sidechains; assistant entries of one
    message.id merged; thinking -> reasoning_content; tool_use -> tool_calls; tool_result -> tool messages."""
    E = [e for e in E if e.get("type") in ("user", "assistant") and not e.get("isSidechain")]
    by = {e.get("uuid"): e for e in E}
    last = E[-1] if E else None
    chain, cur, seen = [], last, set()
    while cur is not None and cur.get("uuid") not in seen:
        seen.add(cur.get("uuid")); chain.append(cur); cur = by.get(cur.get("parentUuid"))
    chain.reverse()
    if len(chain) < len(E) // 2: chain = E
    msgs, tools, models = [], Tools(), {}
    for e in chain:
        m = e.get("message") or {}
        if e["type"] == "user":
            if e.get("isCompactSummary") or e.get("isMeta"): continue
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "tool_result" for p in c):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "tool_result":
                        msgs.append({"role": "tool", "content": scrub(clip(cc_text(p.get("content")), max_tool))})
                continue
            s = cc_text(c)
            if s.strip().startswith(("<command-name>", "<local-command-stdout>", "<command-message>")): continue
            if s.strip(): msgs.append({"role": "user", "content": scrub(s)})
        else:
            if m.get("model") and not str(m["model"]).startswith("<"): models[m["model"]] = models.get(m["model"], 0) + 1
            think, text, calls = [], [], []
            for p in m.get("content") or []:
                if not isinstance(p, dict): continue
                if p.get("type") == "thinking" and (p.get("thinking") or "").strip(): think.append(p["thinking"])
                elif p.get("type") == "text" and p.get("text"): text.append(p["text"])
                elif p.get("type") == "tool_use":
                    args = p.get("input") or {}
                    if not isinstance(args, dict): args = {"input": args}
                    args = json.loads(scrub(json.dumps(args)))
                    calls.append({"type": "function", "function": {"name": p.get("name", "tool"), "arguments": args}})
                    tools.see(p.get("name", "tool"), args)
            if not (think or text or calls): continue
            prev = msgs[-1] if msgs else None
            mid = m.get("id")
            if prev is not None and prev["role"] == "assistant" and mid and prev.get("_mid") == mid:
                if think: prev["reasoning_content"] = (prev.get("reasoning_content", "") + "\n\n" + "\n\n".join(think)).strip()
                if text: prev["content"] = (prev["content"] + "\n\n" + "\n\n".join(text)).strip()
                if calls: prev.setdefault("tool_calls", []).extend(calls)
                continue
            a = {"role": "assistant", "content": scrub("\n\n".join(text)), "_mid": mid}
            if think: a["reasoning_content"] = scrub("\n\n".join(think))
            if calls: a["tool_calls"] = calls
            msgs.append(a)
    for a in msgs: a.pop("_mid", None)
    model = max(models, key=models.get) if models else None
    return msgs, tools, model


def pi_active_path(E):
    by = {e["id"]: e for e in E if "id" in e}
    last = next((e for e in reversed(E) if e.get("type") == "message"), None)
    path, cur = [], last
    while cur is not None:
        path.append(cur); cur = by.get(cur.get("parentId"))
    return list(reversed(path))


def pi_session(E, max_tool):
    model = next((e.get("modelId") for e in E if e.get("type") == "model_change"), None)
    msgs, tools = [], Tools()
    for e in pi_active_path(E):
        t = e.get("type")
        if t == "custom_message":
            msgs.append({"role": "user", "content": scrub(str(e.get("content", "")))}); continue
        if t != "message": continue
        m = e["message"]; r = m.get("role")
        if r == "user":
            s = texts(m.get("content"))
            if s.strip(): msgs.append({"role": "user", "content": scrub(s)})
        elif r == "bashExecution":
            msgs.append({"role": "user", "content": scrub(f"$ {m.get('command', '')}\n{clip(m.get('output', ''), max_tool)}")})
        elif r == "assistant":
            think, text, calls = [], [], []
            for p in m.get("content") or []:
                if p.get("type") == "thinking" and (p.get("thinking") or "").strip(): think.append(p["thinking"])
                elif p.get("type") == "text" and p.get("text"): text.append(p["text"])
                elif p.get("type") == "toolCall":
                    args = p.get("arguments") or {}
                    if not isinstance(args, dict): args = {"input": args}
                    args = json.loads(scrub(json.dumps(args)))
                    calls.append({"type": "function", "function": {"name": p.get("name", "tool"), "arguments": args}})
                    tools.see(p.get("name", "tool"), args)
            if not (think or text or calls): continue
            a = {"role": "assistant", "content": scrub("\n\n".join(text))}
            if think: a["reasoning_content"] = scrub("\n\n".join(think))
            if calls: a["tool_calls"] = calls
            msgs.append(a)
        elif r == "toolResult":
            msgs.append({"role": "tool", "content": scrub(clip(texts(m.get("content")), max_tool))})
    return msgs, tools, model


def turns(msgs):
    """[user msgs + everything up to the next user message] groups."""
    out, cur = [], []
    for m in msgs:
        if m["role"] == "user" and cur and any(x["role"] != "user" for x in cur): out.append(cur); cur = []
        cur.append(m)
    if cur: out.append(cur)
    return out


def merge_assistants(msgs):
    """Consecutive assistant entries (a thinking block, a text block and a tool_use logged as separate entries) become one
    assistant message: reasoning and content joined, tool calls concatenated."""
    out = []
    for m in msgs:
        p = out[-1] if out else None
        if m["role"] == "assistant" and p is not None and p["role"] == "assistant":
            if m.get("reasoning_content"): p["reasoning_content"] = ((p.get("reasoning_content") or "") + "\n\n" + m["reasoning_content"]).strip()
            if m.get("content"): p["content"] = ((p.get("content") or "") + "\n\n" + m["content"]).strip()
            if m.get("tool_calls"): p.setdefault("tool_calls", []).extend(m["tool_calls"])
        else: out.append(dict(m))
    return out


def load_session(path, max_tool=16000):
    """-> (msgs, tools, model, format, tool_info) or (None, reason)"""
    try:
        E = [json.loads(l) for l in open(path, encoding="utf-8", errors="replace") if l.strip().startswith("{")]
    except Exception:
        return None, "parse"
    if not E: return None, "empty"
    if any(e.get("type") == "session" for e in E[:3]):
        msgs, tools, model = pi_session(E, max_tool); fmt = "pi"
    elif any(e.get("type") in ("user", "assistant") for e in E):
        msgs, tools, model = claude_code(E, max_tool); fmt = "claude-code"
    else:
        return None, "format"
    schema, info = tools.schema(fmt)
    return (merge_assistants(msgs), schema, model, fmt, info), None
