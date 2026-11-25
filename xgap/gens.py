"""gens.py -- document generators: stream a source (files shuffled with the source's seed, optional per-file token cap so
every source draws from several shards) and yield documents (adapters.py). `state` is the builder's running counters."""
import glob, json, os, random, sys

from common import *
from adapters import ADAPT
import sessfmt

AGT = os.environ.get("XGAP_AGT_RAW", "/data/xcorp/ds41/agt/raw")      # public agent-trace datasets, pinned snapshots


def list_files(repo, config, split, rev):
    from datasets import load_dataset_builder
    b = load_dataset_builder(repo, config, revision=rev)
    return list(b.config.data_files.get(split, [])) if b.config.data_files else []


def records(spec, rev, rng, state):
    """(record) over the source's files in shuffled order; at most spec['per_file'] tokens accepted per file."""
    from datasets import load_dataset
    repo, config, split = spec["repo"], spec.get("config"), spec.get("split", "train")
    files = spec.get("data_files") or list_files(repo, config, split, rev) or [None]
    files = list(files); rng.shuffle(files)
    for f in files[: spec.get("max_files", 1000)]:
        kw = dict(split=split, streaming=True, revision=rev)
        if f is not None: kw["data_files"] = {split: [f]}
        ds = load_dataset(repo, config, **kw)
        if spec.get("columns"): ds = ds.select_columns(spec["columns"])
        t0 = state.tok
        state.files += 1
        for r in ds:
            yield r
            if spec.get("per_file") and state.tok - t0 >= spec["per_file"]: break


def g_stream(spec, rev, rng, state):
    ad = ADAPT[spec["adapter"]]
    for r in records(spec, rev, rng, state):
        try: res = ad(r, spec)
        except Exception as e:
            state.skip("adapter-error"); continue
        if isinstance(res, tuple): state.skip("adapt:" + res[1]); continue
        for d in res: yield d


def g_oasst(spec, rev, rng, state):
    """OpenAssistant/oasst2: one thread per message tree, following the best-ranked reply at each level."""
    from datasets import load_dataset
    ds = load_dataset(spec["repo"], split="train+validation", revision=rev)
    kids, roots = {}, []
    for r in ds:
        if r["deleted"] or not r["review_result"]: continue
        if r["parent_id"] is None: roots.append(r)
        else: kids.setdefault(r["parent_id"], []).append(r)
    rng.shuffle(roots)
    for root in roots:
        if spec.get("non_english") and root["lang"] == "en": state.skip("english"); continue
        thread, cur = [root], root
        while cur["message_id"] in kids:
            ch = sorted(kids[cur["message_id"]], key=lambda x: (x["rank"] is None, x["rank"] if x["rank"] is not None else 0))
            cur = ch[0]; thread.append(cur)
        msgs = [{"role": "user" if m["role"] == "prompter" else "assistant", "content": m["text"]} for m in thread]
        while msgs and msgs[-1]["role"] != "assistant": msgs.pop()
        if len(msgs) < 2: state.skip("shape"); continue
        yield {"kind": "chat", "msgs": msgs, "tools": None, "category": spec["category"], "lic": "apache-2.0",
               "lang": root["lang"], "extra": {"turns": len(msgs), "tree": root["message_tree_id"]}}


def g_sessions(spec, rev, rng, state):
    """Public agent-trace logs: one 'session' document per file, turns deduplicated across files (forks, continuations)."""
    import hashlib
    files = sorted(glob.glob(os.path.join(AGT, spec["dir"], "**", "*.jsonl"), recursive=True))
    rng.shuffle(files)
    seen = set()
    for f in files:
        res, why = sessfmt.load_session(f)
        if res is None: state.skip(f"session-{why}"); continue
        msgs, tools, model, fmt = res
        keep = []
        for t in sessfmt.turns(msgs):
            h = hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
            if h in seen: continue
            seen.add(h); keep.append(t)
        if not keep or not any(m["role"] == "assistant" for t in keep for m in t): state.skip("session-empty"); continue
        yield {"kind": "session", "turns": keep, "tools": tools, "category": spec["category"], "lic": spec["lic"], "lang": "en",
               "extra": {"session": os.path.basename(f), "model": model, "format": fmt}}


GENS = {"stream": g_stream, "oasst": g_oasst, "sessions": g_sessions}
