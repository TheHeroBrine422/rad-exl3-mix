"""adapters.py -- dataset record -> xgap documents.
A document is a dict: kind "text" {text} | "chat" {msgs, tools}, plus category / lic (record licence) / lang / extra.
Adapters return a list of documents or (None, reason). Chat adapters keep the source's REAL system prompt (no invented
ones); `has_system` is computed from the final messages (first message role system with text)."""
import ast, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xcorp"))
import build as xb          # noqa: E402  (xcorp/build.py: _tc, _tools, _call, TOOLCALL, ad_swe, split_think)
from common import licence_ok  # noqa: E402


def meta_of(r):
    m = r.get("metadata")
    if isinstance(m, str):
        try: m = json.loads(m)
        except Exception:
            try: m = ast.literal_eval(m)
            except Exception: m = {}
    return m if isinstance(m, dict) else {}


# ------------------------------------------------------------------ text
def text_doc(text, category, lic=None, lang=None, **extra):
    if not text or not text.strip(): return None, "empty"
    if lic is not None and not licence_ok(lic): return None, "licence"
    return [{"kind": "text", "text": text, "category": category, "lic": lic, "lang": lang, "extra": extra}]


def ad_commonpile(r, spec):
    """common-pile *_filtered: text + metadata (licence per record)."""
    m = meta_of(r)
    lic = m.get("license") or m.get("licence")
    if spec.get("english_only") and m.get("language") not in (None, "en", "eng"): return None, "lang"
    title = m.get("title")
    return text_doc(r.get("text"), spec["category"], lic, "en", title=title)


def ad_pg19(r, spec):
    return text_doc(r.get("text"), spec["category"], "public-domain (PG-19, apache-2.0 repack)", "en", title=r.get("short_book_title"))


def ad_wikipedia(r, spec):
    t = r.get("text") or ""
    return text_doc(f"{r.get('title', '')}\n\n{t}", spec["category"], "cc-by-sa-3.0 / gfdl", spec.get("lang"), title=r.get("title"))


def ad_fineweb2(r, spec):
    try: sc = float(r.get("language_score") or 1)
    except Exception: sc = 1.0
    if sc < spec.get("min_lang_score", 0.9): return None, "lang-score"
    return text_doc(r.get("text"), spec["category"], "odc-by", spec.get("lang"))


def ad_stack(r, spec):
    c = r.get("content") or ""
    lic = r.get("max_stars_repo_licenses")
    if not c.strip(): return None, "empty"
    return text_doc(f"# {r.get('max_stars_repo_path', '')}\n{c}", spec["category"], str(lic), spec["lang"], repo=r.get("max_stars_repo_name"))


def ad_stackedu(r, spec):
    m = meta_of(r)
    if spec.get("lang") and m.get("language") != spec["lang"]: return None, "lang"
    if m.get("license_type") not in (None, "permissive"): return None, "licence"
    return text_doc(r.get("text"), spec["category"], str(m.get("license") or m.get("detected_licenses")), m.get("language"), repo=m.get("repo_name"))


# ------------------------------------------------------------------ chat
def chat_doc(msgs, tools, category, lic, lang, **extra):
    return [{"kind": "chat", "msgs": msgs, "tools": tools, "category": category, "lic": lic, "lang": lang, "extra": extra}]


def ad_aya(r, spec):
    if r.get("language_code") == "eng" and spec.get("non_english", True): return None, "english"
    q, a = (r.get("inputs") or "").strip(), (r.get("targets") or "").strip()
    if not q or not a: return None, "empty"
    return chat_doc([{"role": "user", "content": q}, {"role": "assistant", "content": a}], None, spec["category"], "apache-2.0", r.get("language_code"))


def ad_wildchat(r, spec):
    if r.get("toxic") or r.get("redacted"): return None, "toxic-or-redacted"
    lang = r.get("language") or ""
    if spec.get("non_english") and lang.lower() == "english": return None, "english"
    conv = r.get("conversation") or []
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in conv if m.get("role") in ("user", "assistant")]
    if len(msgs) < 2 or msgs[0]["role"] != "user": return None, "shape"
    if sum(len(m["content"]) for m in msgs if m["role"] == "assistant") < 100: return None, "thin"
    if len(msgs) < spec.get("min_msgs", 2): return None, "few-turns"
    return chat_doc(msgs, None, spec["category"], "odc-by", lang, turns=len(msgs))


# --- tools
HERMES_FORMAT_HEAD = "You are a function calling AI model"


def hermes_system(v):
    """Hermes' function-calling system prompt is pure format boilerplate (tool list + its own call format) that the
    Qwen template replaces: dropped. Anything else (e.g. the JSON-mode prompt with its schema) is the real prompt: kept."""
    return None if v.strip().startswith(HERMES_FORMAT_HEAD) else v.strip()


def ad_hermes(r, spec):
    raw = r.get("conversations")
    conv = json.loads(raw) if isinstance(raw, str) else raw
    tr = r.get("tools")
    try: tools = xb._tools(json.loads(tr) if isinstance(tr, str) else (tr or []))
    except Exception: return None, "malformed"
    msgs = []
    for m in conv or []:
        f, v = m["from"], m["value"]
        if f == "system":
            s = hermes_system(v)
            if s: msgs.append({"role": "system", "content": s})
        elif f == "human": msgs.append({"role": "user", "content": v})
        elif f == "gpt":
            calls = xb.TOOLCALL.findall(v)
            if "<tool_call>" in v and len(calls) != v.count("<tool_call>"): return None, "malformed"
            if calls:
                try: tc = [xb._call(c) for c in calls]
                except Exception: return None, "malformed"
                msgs.append({"role": "assistant", "content": xb.TOOLCALL.sub("", v).strip(), "tool_calls": tc})
            else: msgs.append({"role": "assistant", "content": v})
        elif f == "tool": msgs.append({"role": "tool", "content": re.sub(r"</?tool_response>", "", v).strip()})
    body = [m for m in msgs if m["role"] != "system"]
    if not body or body[0]["role"] != "user" or not any(m["role"] == "assistant" for m in body): return None, "shape"
    if spec.get("need_tools", True) and not tools: return None, "no-tools"
    return chat_doc(msgs, tools or None, spec["category"], "apache-2.0", "en", subcategory=r.get("subcategory"))


def ad_glaive(r, spec):
    sysm = r.get("system") or ""
    objs, dec, i = [], json.JSONDecoder(), sysm.find("{")
    while i >= 0:
        try:
            o, j = dec.raw_decode(sysm, i); objs.append(o); i = sysm.find("{", j)
        except Exception: i = sysm.find("{", i + 1)
    tools = xb._tools([o for o in objs if isinstance(o, dict) and "name" in o])
    head = sysm.split("{", 1)[0].strip() if tools else sysm.strip()
    head = re.sub(r"^SYSTEM:\s*", "", head).rstrip(" -:\n").strip()
    msgs = [{"role": "system", "content": head}] if head else []
    toks = re.split(r"(USER: |ASSISTANT: |FUNCTION RESPONSE: )", r.get("chat") or "")
    for k in range(1, len(toks) - 1, 2):
        tag, body = toks[k], toks[k + 1].replace("<|endoftext|>", "").strip()
        if tag == "USER: ": msgs.append({"role": "user", "content": body})
        elif tag == "FUNCTION RESPONSE: ": msgs.append({"role": "tool", "content": body})
        else:
            if "<functioncall>" in body:
                pre, _, call = body.partition("<functioncall>")
                try:
                    c = json.loads(call.strip().replace("'{", "{").replace("}'", "}"))
                    msgs.append({"role": "assistant", "content": pre.strip(), "tool_calls": [xb._tc(c["name"], c.get("arguments", {}))]})
                except Exception: return None, "tool-call"
            else: msgs.append({"role": "assistant", "content": body})
    body = [m for m in msgs if m["role"] != "system"]
    if not body or body[0]["role"] != "user" or not any(m["role"] == "assistant" for m in body): return None, "shape"
    if any(m["role"] == "tool" for m in msgs) and not tools: return None, "no-tools"
    return chat_doc(msgs, tools or None, spec["category"], "apache-2.0", "en")


def ad_when2call(r, spec):
    tl = r.get("tools") or []
    tools = xb._tools(tl)
    msgs = []
    for m in r.get("messages") or []:
        c = m.get("content") or ""
        if m["role"] == "assistant" and "<TOOLCALL>" in c:
            try: calls = json.loads(c.split("<TOOLCALL>")[1].split("</TOOLCALL>")[0])
            except Exception: return None, "malformed"
            msgs.append({"role": "assistant", "content": "", "tool_calls": [xb._tc(x["name"], x.get("arguments", {})) for x in calls]})
        else: msgs.append({"role": m["role"], "content": c})
    body = [m for m in msgs if m["role"] != "system"]
    if not tools or not body or body[0]["role"] != "user": return None, "shape"
    return chat_doc(msgs, tools, spec["category"], "cc-by-4.0", "en")


def ad_swe(r, spec):
    res = xb.ad_swe(r)
    if res[0] is None: return res
    lic = r.get("license") or spec.get("lic")
    if lic and not licence_ok(lic): return None, "licence"
    return chat_doc(res[1], res[2], spec["category"], lic, "en", repo=r.get("repo"), tools_inferred=False)


ADAPT = {"commonpile": ad_commonpile, "pg19": ad_pg19, "wikipedia": ad_wikipedia, "fineweb2": ad_fineweb2,
         "stack": ad_stack, "stackedu": ad_stackedu, "aya": ad_aya, "wildchat": ad_wildchat, "hermes": ad_hermes,
         "glaive": ad_glaive, "when2call": ad_when2call, "swe": ad_swe}
