#!/usr/bin/env python3
"""build.py SOURCE OUTDIR [--mult M] [--budget TOKENS] [--seed S] -- one source of the extended corpus ("xcorp"): the
challenge corpus's own upstream sources (StillDeadcode/qwen3.8-27b-quant-challenge corpus/manifest.json: the same
datasets at the same revisions, categories, kinds and per-source token caps), M times its per-split budget (default
12: ~100M tokens over all sources). For proxy research: Hessians, YAQA, QAD on small models. Not the agentic
traces (generated for the challenge, not published).

Kept apart from the challenge's documents:
  * the head of every stream is skipped (the challenge read only the first few thousand records of each source):
    multi-file sources start at the second half of their files; single-file sources skip 1.1 x the
    challenge's scanned count (all it read);
  * every candidate is checked line by line against calib + public (index.py): rejected when >= 20% of its long lines
    (>= 2) occur there, template boilerplate (lines in >= 3 calib documents) not counted. The private split is not
    published; the head skip is the guard there. Before anything trained on xcorp ships, check the challenge's rules.

Rendering as the challenge: text = the source text (long ones a random window of the cap); chat = Qwen3.8's chat
template (its xhigh preamble, tools in the system block, reasoning in reasoning_content, kept on every turn); a chat
over the cap loses trailing turns. Output OUTDIR/<slug>.jsonl lines {source, category, kind, tokens, prompt} and
OUTDIR/<slug>.stats.json."""
import argparse, hashlib, json, os, random, re, sys, time

MANIFEST = os.environ.get("XCORP_MANIFEST", "/data/xcorp/ref/corpus/manifest.json")
TOKENIZER = os.environ.get("XCORP_TOKENIZER", "/data/models/qwen38-27b-bf16")
INDEX = os.environ.get("XCORP_INDEX", "/data/xcorp/ref/index.json")
SKIP_FACTOR = 1.1   # the challenge read the first `scanned` records (all three splits); a larger skip ran small
                    # datasets dry (s1k: 1k rows; hermes func_calling's last ~300 are a different, malformed task)

# source -> (repo, config, split, adapter, file part). Revisions from the manifest.
REPO = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-10", "train", "text", "late"),
    "wikimedia": ("common-pile/wikimedia_filtered", None, "train", "wikimedia", "late"),
    "arxiv": ("common-pile/arxiv_papers_filtered", None, "train", "text", "late"),
    "arxiv-long": ("common-pile/arxiv_papers_filtered", None, "train", "text", "late2"),
    "peS2o": ("common-pile/peS2o_filtered", None, "train", "text", "late"),
    "pubmed": ("common-pile/pubmed_filtered", None, "train", "text", "late"),
    "finemath-4plus": ("HuggingFaceTB/finemath", "finemath-4plus", "train", "text", "late"),
    "libretexts": ("common-pile/libretexts_filtered", None, "train", "text", "late"),
    "pg19": ("emozilla/pg19", None, "train", "book", "late"),
    "pg19-long": ("emozilla/pg19", None, "train", "book", "late2"),
    "gutenberg": ("common-pile/project_gutenberg_filtered", None, "train", "book", "late"),
    "doab": ("common-pile/doab_filtered", None, "train", "book", "late"),
    "caselaw": ("common-pile/caselaw_access_project_filtered", None, "train", "text", "late"),
    "stackexchange": ("common-pile/stackexchange_filtered", None, "train", "text", "late"),
    "stackv2-edu": ("common-pile/stackv2_edu_filtered", None, "train", "text", "late"),
    "the-stack-smol-xl": ("bigcode/the-stack-smol-xl", None, "train", "stack", "late"),
    "wildchat": ("allenai/WildChat-4.8M", None, "train", "wildchat", "late"),
    "smoltalk2/everyday": ("HuggingFaceTB/smoltalk2", "SFT", "smoltalk_everyday_convs_reasoning_Qwen3_32B_think", "smoltalk", "skip"),
    "smoltalk2/systemchats": ("HuggingFaceTB/smoltalk2", "SFT", "smoltalk_systemchats_Qwen3_32B_think", "smoltalk", "skip"),
    "smoltalk2/multilingual": ("HuggingFaceTB/smoltalk2", "SFT", "smoltalk_multilingual8_Qwen3_32B_think", "smoltalk", "skip"),
    "smoltalk2/multi-turn-reasoning": ("HuggingFaceTB/smoltalk2", "SFT", "multi_turn_reasoning_if_think", "smoltalk", "skip"),
    "smoltalk2/table-gpt": ("HuggingFaceTB/smoltalk2", "SFT", "table_gpt_Qwen3_32B_think", "smoltalk", "skip"),
    "smoltalk2/s1k": ("HuggingFaceTB/smoltalk2", "SFT", "s1k_1.1_think", "smoltalk", "skip"),
    "smoltalk2/longalign": ("HuggingFaceTB/smoltalk2", "SFT", "LongAlign_64k_Qwen3_32B_yarn_131k_think", "smoltalk", "skip"),
    "openthoughts3": ("open-thoughts/OpenThoughts3-1.2M", None, "train", "sharegpt", "late"),
    "hermes-fc": ("NousResearch/hermes-function-calling-v1", "func_calling", "train", "hermes", "skip"),
    "glaive-fc-v2": ("glaiveai/glaive-function-calling-v2", None, "train", "glaive", "skip"),
    "when2call": ("nvidia/When2Call", "train_sft", "train", "when2call", "skip"),
    "swe-rebench-openhands": ("nebius/SWE-rebench-openhands-trajectories", None, "train", "swe", "late"),
    "nemotron-swe": ("nvidia/Nemotron-SWE-v1", None, "r2e_gym", "swe", "late"),
    "swe-smith": ("SWE-bench/SWE-smith-trajectories", None, "tool", "swe", "late"),
}
REV_OF = {"fineweb-edu": "HuggingFaceFW/fineweb-edu", "fineweb2-hq": "epfml/FineWeb2-HQ", "wikipedia": "wikimedia/wikipedia"}


def repo_for(src):
    if src.startswith("fineweb2-hq/"): return ("epfml/FineWeb2-HQ", src.split("/")[1], "train", "text", "late")
    if src.startswith("wikipedia/"): return ("wikimedia/wikipedia", "20231101." + src.split("/")[1], "train", "wikipedia", "late")
    return REPO[src]


# ------------------------------------------------------------------ the stream, past the head
def stream(repo, config, split, part, rev, scanned, seed):
    import datasets
    from datasets import load_dataset, load_dataset_builder
    b = load_dataset_builder(repo, config, revision=rev)
    files = list(b.config.data_files.get(split, [])) if b.config.data_files else []
    kw = dict(split=split, streaming=True, revision=rev)
    if part in ("late", "late2") and len(files) >= 4:
        rng = random.Random(seed)
        lo = len(files) // 2 if part == "late" else (3 * len(files)) // 4
        tail = files[lo:]; rng.shuffle(tail)
        ds = load_dataset(repo, config, data_files={split: tail}, **kw)
        return ds, f"files {lo}..{len(files)} of {len(files)} (shuffled)"
    ds = load_dataset(repo, config, **kw)
    n = int(SKIP_FACTOR * max(scanned, 50))
    return ds.skip(n), f"skip first {n} records ({len(files)} files)"


# ------------------------------------------------------------------ filters
def long_lines(text):
    return [l.strip() for l in text.split("\n") if len(l.strip()) >= 50]


def lhash(l): return hashlib.blake2b(l.encode(), digest_size=8).hexdigest()


def repetitive(text):
    ls = [l for l in text.split("\n") if l.strip()]
    if len(ls) >= 10 and len(set(ls)) / len(ls) < 0.5: return True
    w = text.split()
    return len(w) >= 200 and len(set(w)) / len(w) < 0.1


def prose(text):   # books: mostly sentences of letters, not tables / front matter
    if not text: return False
    letters = sum(c.isalpha() for c in text) / len(text)
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines: return False
    short = sum(len(l) < 30 for l in lines) / len(lines)
    return letters > 0.7 and short < 0.6


# ------------------------------------------------------------------ adapters: record -> ("text", str) | ("chat", messages, tools) | (None, reason)
THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def split_think(m):
    c = m.get("content") or ""
    g = THINK.match(c)
    if g:
        m = dict(m); m["reasoning_content"] = g.group(1).strip("\n"); m["content"] = c[g.end():]
    return m


def as_text(r, field="text"):
    t = r.get(field) or ""
    return ("text", t) if t.strip() else (None, "empty")


def ad_wikimedia(r):
    meta = r.get("metadata") or {}
    url = (meta.get("url") or meta.get("source") or r.get("source") or "") if isinstance(meta, dict) else ""
    if "wikipedia" not in str(url).lower() and "wikipedia" not in str(r.get("source", "")).lower(): return None, "filter"
    return as_text(r)


def ad_wikipedia(r):
    t = r.get("text") or ""
    return ("text", f"{r.get('title', '')}\n\n{t}") if t.strip() else (None, "empty")


def ad_stack(r):
    c = r.get("content") or ""
    if not c.strip(): return None, "empty"
    if r.get("lang") in ("JSON", "CSV", "SVG", "XML", "YAML", "Text", "TSV"): return None, "lang"
    return "text", f"# {r.get('max_stars_repo_path', '')}\n{c}"


def ad_book(r):
    t = as_text(r)
    return t


def ad_wildchat(r):
    if r.get("toxic") or r.get("redacted"): return None, "toxic-or-redacted"
    conv = r.get("conversation") or []
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in conv if m.get("role") in ("user", "assistant")]
    if len(msgs) < 2 or msgs[0]["role"] != "user": return None, "shape"
    if sum(len(m["content"]) for m in msgs if m["role"] == "assistant") < 200: return None, "thin"
    return "chat", msgs, None


def ad_smoltalk(r):
    msgs = [split_think(dict(role=m["role"], content=m.get("content") or "")) for m in r.get("messages") or []]
    kw = r.get("chat_template_kwargs") or {}
    ci = kw.get("custom_instructions") if isinstance(kw, dict) else None
    if ci and not any(m["role"] == "system" for m in msgs): msgs = [{"role": "system", "content": ci}] + msgs
    if not any(m["role"] == "assistant" for m in msgs): return None, "shape"
    return "chat", msgs, None


def ad_sharegpt(r):
    role = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
    msgs = [split_think({"role": role.get(m["from"], "user"), "content": m["value"]}) for m in r.get("conversations") or []]
    if not any(m.get("reasoning_content") for m in msgs if m["role"] == "assistant"): return None, "no-think"
    return "chat", msgs, None


def _tc(name, args):
    if isinstance(args, str):
        try: args = json.loads(args)
        except Exception: args = {"input": args}
    if not isinstance(args, dict): args = {"input": args}
    return {"type": "function", "function": {"name": name, "arguments": args}}


def _tools(lst):
    out = []
    for t in lst:
        if isinstance(t, str):
            try: t = json.loads(t)
            except Exception: continue
        if "function" not in t: t = {"type": "function", "function": t}
        out.append(t)
    return out


TOOLCALL = re.compile(r"<tool_call>(?:\s|\\n)*(\{.*?\})(?:\s|\\n)*</tool_call>", re.S)   # some records: a literal \n


def _call(c):   # one call's JSON; some records write Python literals (single quotes)
    import ast
    try: o = json.loads(c)
    except Exception: o = ast.literal_eval(c)
    return _tc(o["name"], o.get("arguments", {}))


def ad_hermes(r):
    try: tools = _tools(json.loads(r.get("tools") or "[]"))
    except Exception: return None, "malformed"
    msgs = []
    for m in r.get("conversations") or []:
        f, v = m["from"], m["value"]
        if f == "system": continue                         # its tool list is the template's now
        if f == "human": msgs.append({"role": "user", "content": v})
        elif f == "gpt":
            calls = TOOLCALL.findall(v)
            if "<tool_call>" in v and len(calls) != v.count("<tool_call>"): return None, "malformed"
            if calls:
                try: tc = [_call(c) for c in calls]
                except Exception: return None, "malformed"
                msgs.append({"role": "assistant", "content": TOOLCALL.sub("", v).strip(), "tool_calls": tc})
            else: msgs.append({"role": "assistant", "content": v})
        elif f == "tool":
            body = re.sub(r"</?tool_response>", "", v).strip()
            msgs.append({"role": "tool", "content": body})
    if not tools or not msgs or msgs[0]["role"] != "user": return None, "shape"
    return "chat", msgs, tools


def ad_glaive(r):
    sysm = r.get("system") or ""
    objs, dec, i = [], json.JSONDecoder(), sysm.find("{")
    while i >= 0:
        try:
            o, j = dec.raw_decode(sysm, i); objs.append(o); i = sysm.find("{", j)
        except Exception: i = sysm.find("{", i + 1)
    tools = _tools([o for o in objs if isinstance(o, dict) and "name" in o])
    if not tools: return None, "no-tools"
    msgs = []
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
                    msgs.append({"role": "assistant", "content": pre.strip(), "tool_calls": [_tc(c["name"], c.get("arguments", {}))]})
                except Exception: return None, "tool-call"
            else: msgs.append({"role": "assistant", "content": body})
    if not msgs or msgs[0]["role"] != "user" or not any(m["role"] == "assistant" for m in msgs): return None, "shape"
    return "chat", msgs, tools


def ad_when2call(r):
    tools = _tools(r.get("tools") or [])
    msgs = []
    for m in r.get("messages") or []:
        c = m.get("content") or ""
        if m["role"] == "assistant" and "<TOOLCALL>" in c:
            try: calls = json.loads(c.split("<TOOLCALL>")[1].split("</TOOLCALL>")[0])
            except Exception: return None, "malformed"
            msgs.append({"role": "assistant", "content": "", "tool_calls": [_tc(x["name"], x.get("arguments", {})) for x in calls]})
        else: msgs.append({"role": m["role"], "content": c})
    if not tools or not msgs: return None, "shape"
    return "chat", msgs, tools


def ad_swe(r):
    raw = r.get("messages") or r.get("trajectory") or "[]"
    try:
        msgs0 = json.loads(raw) if isinstance(raw, str) else raw
        tools = _tools(json.loads(r["tools"]) if isinstance(r.get("tools"), str) else (r.get("tools") or []))
    except Exception: return None, "malformed"
    msgs = []
    for m in msgs0:
        c = m.get("content")
        if isinstance(c, list): c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        c = c or ""
        role = m.get("role")
        if role == "assistant":
            tc = []
            for t in m.get("tool_calls") or []:
                f = t.get("function", t); tc.append(_tc(f.get("name"), f.get("arguments", {})))
            mm = split_think({"role": "assistant", "content": c})
            if tc: mm["tool_calls"] = tc
            msgs.append(mm)
        elif role in ("user", "system", "tool"): msgs.append({"role": role, "content": c})
    if not any(m["role"] == "assistant" for m in msgs): return None, "shape"
    # structured tool calls only: some files of swe-smith's "tool" split hold the old text form (observations as user
    # turns, calls as bare text), which the challenge's documents never are
    if not any(m.get("tool_calls") for m in msgs) or not any(m["role"] == "tool" for m in msgs): return None, "no-tool-calls"
    return "chat", msgs, tools or None


ADAPT = {"text": as_text, "wikimedia": ad_wikimedia, "wikipedia": ad_wikipedia, "stack": ad_stack, "book": ad_book,
         "wildchat": ad_wildchat, "smoltalk": ad_smoltalk, "sharegpt": ad_sharegpt, "hermes": ad_hermes,
         "glaive": ad_glaive, "when2call": ad_when2call, "swe": ad_swe}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("out")
    ap.add_argument("--mult", type=float, default=12); ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261102); ap.add_argument("--max-scan", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    man = json.load(open(MANIFEST))["datasets"]
    S = man["sources"][a.source]
    repo, config, split, adname, part = repo_for(a.source)
    rev = man["revisions"][repo]
    budget = a.budget or int(S["budget_per_split"] * a.mult)
    cap, cat, kind = S["max_tokens"], S["category"], S["kind"]
    idx = json.load(open(INDEX)); bad = set(idx["lines"]); idx["boiler"] = set(idx["boiler"])
    seed = int(hashlib.sha256(f"{a.seed}:{a.source}".encode()).hexdigest()[:8], 16); rng = random.Random(seed)
    ds, how = stream(repo, config, split, part, rev, S["stats"]["scanned"], seed)
    slug = a.source.replace("/", "__")
    os.makedirs(a.out, exist_ok=True)
    fo = open(f"{a.out}/{slug}.jsonl.part", "w")
    st = {"source": a.source, "repo": repo, "config": config, "split": split, "revision": rev, "stream": how,
          "budget": budget, "cap": cap, "scanned": 0, "accepted": 0, "tokens": 0, "truncated": 0}
    seen = set(); t0 = time.time()
    def skip(why): st[f"skip:{why}"] = st.get(f"skip:{why}", 0) + 1
    for r in ds:
        if st["tokens"] >= budget or (a.max_scan and st["scanned"] >= a.max_scan): break
        st["scanned"] += 1
        res = ADAPT[adname](r)
        if res[0] is None: skip(res[1]); continue
        if res[0] == "text":
            text = res[1]
            if adname == "book" and not prose(text[:20000]): skip("not-prose"); continue
            if repetitive(text[:50000]): skip("repetitive"); continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) < (2000 if a.source.endswith("-long") else 128): skip("short"); continue
            if len(ids) > cap:
                s0 = rng.randrange(0, len(ids) - cap + 1); ids = ids[s0:s0 + cap]; text = tok.decode(ids); st["truncated"] += 1
            prompt = text
        else:
            _, msgs, tools = res
            try: prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
            except Exception: skip("render"); continue
            n = len(tok(prompt, add_special_tokens=False)["input_ids"])
            while n > cap and len(msgs) > 2:
                msgs = msgs[:-1]
                while msgs and msgs[-1]["role"] != "assistant": msgs = msgs[:-1]
                if not msgs: break
                prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
                n = len(tok(prompt, add_special_tokens=False)["input_ids"]); st["truncated"] += 1
            if not msgs or n > cap: skip("too-long"); continue
            if n < 64: skip("short"); continue
        h = hashlib.sha256(prompt.encode()).hexdigest()
        if h in seen: skip("duplicate"); continue
        ll = [l for l in long_lines(prompt) if lhash(l) not in idx["boiler"]]
        hit = sum(lhash(l) in bad for l in ll)
        if hit >= 2 and hit >= 0.2 * len(ll): skip("in-challenge"); continue
        seen.add(h)
        ntok = len(tok(prompt, add_special_tokens=False)["input_ids"])
        fo.write(json.dumps({"source": a.source, "category": cat, "kind": kind, "tokens": ntok, "prompt": prompt},
                            ensure_ascii=False) + "\n")
        st["accepted"] += 1; st["tokens"] += ntok
        if st["accepted"] % 200 == 0:
            print(f"{a.source}: {st['tokens']}/{budget} tokens, {st['accepted']} docs of {st['scanned']} scanned, "
                  f"{time.time() - t0:.0f}s", flush=True)
    fo.close(); os.replace(f"{a.out}/{slug}.jsonl.part", f"{a.out}/{slug}.jsonl")
    st["seconds"] = round(time.time() - t0, 1)
    json.dump(st, open(f"{a.out}/{slug}.stats.json", "w"), indent=1)
    print("DONE", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # datasets' streaming prefetch threads otherwise keep the process alive after the source is done
