#!/usr/bin/env python3
"""build_source.py SOURCE_ID [--scale S] [--max-seconds N] -- one xgap-v1 source -> work/src/<id>.jsonl (+ .meta.jsonl,
.stats.json). Restartable: a finished source (stats file present) is skipped; output is written .part then renamed.
Budgets are the registry's (sources.py) times --scale; a document is accepted while its budget key (length band /
language / cell) is below budget. Rendering, dedup and tags: see README.md."""
import argparse, collections, glob, json, os, random, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import sources, gens

DATASET_LICENCE = {
    "emozilla/pg19": "public domain texts (PG-19, apache-2.0 repack)", "wikimedia/wikipedia": "cc-by-sa-3.0 / gfdl",
    "epfml/FineWeb2-HQ": "odc-by", "HuggingFaceFW/fineweb-2": "odc-by", "bigcode/the-stack-smol-xl": "permissive (per-file licences)",
    "common-pile/stackv2_edu_filtered": "permissive (per-file licences)", "CohereLabs/aya_dataset": "apache-2.0",
    "OpenAssistant/oasst2": "apache-2.0", "allenai/WildChat-4.8M": "odc-by", "glaiveai/glaive-function-calling-v2": "apache-2.0",
    "NousResearch/hermes-function-calling-v1": "apache-2.0", "nvidia/When2Call": "cc-by-4.0",
    "nebius/SWE-rebench-openhands-trajectories": "cc-by-4.0", "nvidia/Nemotron-SWE-v1": "cc-by-4.0 (per-record licence)",
    "SWE-bench/SWE-smith-trajectories": "mit"}
XL_TOK = 131072


class State:
    def __init__(self): self.tok = 0; self.files = 0; self.skips = collections.Counter()
    def skip(self, why): self.skips[why] += 1


def revision(spec):
    if spec["gen"] == "sessions":
        t = glob.glob(os.path.join(gens.AGT, spec["dir"], ".cache", "huggingface", "trees", "*.json"))
        return os.path.basename(t[0])[:-5] if t else None
    os.makedirs(f"{ROOT}/work/rev", exist_ok=True)
    p = f"{ROOT}/work/rev/{spec['repo'].replace('/', '__')}.txt"
    if os.path.exists(p): return open(p).read().strip()
    from huggingface_hub import HfApi
    sha = HfApi().dataset_info(spec["repo"]).sha
    open(p + f".{os.getpid()}", "w").write(sha); os.replace(p + f".{os.getpid()}", p)
    return sha


def window(text, n, max_tok, rng):
    """A contiguous passage of <= max_tok tokens starting and ending at paragraph breaks (books longer than 128k)."""
    per = len(text) / n
    target = int(max_tok * 0.96 * per)
    for _ in range(12):
        lo = rng.randrange(0, max(1, len(text) - target))
        s = text.find("\n\n", lo); s = 0 if s < 0 or s > lo + 20000 else s + 2
        e = text.rfind("\n\n", s + target // 2, s + target)
        piece = text[s:(e if e > 0 else s + target)].strip("\n")
        m = ntok(piece)
        if m <= max_tok: return piece, m
        target = int(target * 0.95)
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid"); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pilot", action="store_true", help="always take the first document of every budget key")
    ap.add_argument("--max-seconds", type=int, default=3000); ap.add_argument("--max-scan", type=int, default=3_000_000)
    a = ap.parse_args()
    spec = dict(sources.BY_ID[a.sid]); sid = a.sid
    if finished(sid): print("finished, skipping", sid); return
    if spec.get("stackedu_lang"):
        m = json.load(open(f"{ROOT}/work/stackedu_langs.json"))
        spec["data_files"] = [k for k, v in sorted(m.items()) if v == spec["stackedu_lang"]]
    rev = revision(spec)
    budgets = {k: max(1, int(v * a.scale)) for k, v in spec["budgets"].items()}
    used = collections.Counter(); streak = collections.Counter(); done = set()
    idx = load_index()
    rng = random.Random(int(hashlib_seed(sid), 16))
    st = State(); out = Out(sid); seen = set(); t0 = time.time()
    scanned = 0; dups = collections.Counter(); by_band = collections.Counter(); n_acc = 0
    dlic = DATASET_LICENCE.get(spec["repo"], "openly licensed (Common Pile; per-record licence)" if spec["repo"].startswith("common-pile/") else None)

    def accept(prompt, n, doc, key, cell, think=None, has_system=None, length=None, extra=None):
        nonlocal n_acc
        if key not in budgets: st.skip(f"no-budget:{key}"); return False
        # a document may cross the budget by at most half its size; the first document of a key is always taken in --pilot
        # runs (budgets there are a few thousand tokens), otherwise only when it is at most twice the budget
        if (used[key] > 0 or not (a.pilot or n <= 2 * budgets[key])) and used[key] + n / 2 > budgets[key]:
            st.skip(f"full:{key}"); streak[key] += 1
            if streak[key] >= 40: done.add(key)                     # 40 in a row bounced: the key is as full as it gets
            return False
        streak[key] = 0
        d = idx.check(prompt)
        if d: dups[f"{d[0]}:{d[1]}"] += 1; return False
        h = norm_hash(prompt)
        if h in seen: dups["xgap-in-source"] += 1; return False
        seen.add(h)
        did = "xg-" + h[:14]
        line = {"id": did, "source": spec["source"], "category": spec["category"], "kind": doc["kind"] if doc["kind"] != "session" else "chat",
                "tokens": n, "prompt": prompt}
        meta = {"id": did, "origin": "xgap-v1", "cell": cell, "category": spec["category"], "source": spec["source"], "dataset": spec["repo"],
                "config": spec.get("config"), "split": spec.get("split", "train"), "revision": rev, "licence": doc.get("lic") or spec.get("lic") or dlic,
                "kind": line["kind"], "tokens": n, "length_band": band(n), "lang": doc.get("lang"), "src_id": sid, "budget_key": key}
        if think: meta["thinking"] = think
        if has_system is not None: meta["has_system"] = has_system
        if length: meta["length"] = length
        meta.update({k: v for k, v in (doc.get("extra") or {}).items() if v is not None}); meta.update(extra or {})
        out.write(line, meta)
        used[key] += n; st.tok += n; n_acc += 1; by_band[band(n)] += n
        return True

    def keyer(n):
        kt = spec["keyer"]
        if kt == "all": return "all", spec["cell"]
        if kt == "band":
            b = band(n); return (b, spec["cell"]) if b in KS else (None, None)
        if n < 8192: return "cl", "code-langs"                  # kt == "code"
        b = band(n); return (b, "long-docs") if b in KS else (None, None)

    def do_text(doc):
        text = doc["text"]
        if has_special(text): st.skip("special-token"); return
        if repetitive(text[:50000]): st.skip("repetitive"); return
        extra = {}
        if spec.get("window") and len(text) > 700_000: text = text[: 700_000]; extra["pre_cut"] = True
        n = ntok(text)
        if n < spec["min_tok"]: st.skip("short"); return
        if n > spec["max_tok"]:
            if not spec.get("window"): st.skip("too-long"); return
            orig = n
            text, n = window(text, n, spec["max_tok"], rng)
            if text is None: st.skip("window-fail"); return
            extra.update(windowed=True, orig_tokens=orig)
        key, cell = keyer(n)
        if key is None: st.skip("band"); return
        accept(text, n, doc, key, cell, extra=extra)

    def render(msgs, tools):
        try: return render_chat(msgs, tools)
        except Exception: st.skip("render"); return None, None

    def has_sys(msgs): return bool(msgs and msgs[0]["role"] == "system" and (msgs[0].get("content") or "").strip())

    def do_chat(doc):
        msgs = doc["msgs"]
        if has_special(json.dumps(msgs, ensure_ascii=False)): st.skip("special-token"); return
        prompt, think = render(msgs, doc["tools"])
        if prompt is None: return
        n = ntok(prompt)
        if n < spec["min_tok"]: st.skip("short"); return
        length = None; key = "all"
        if n > spec["max_tok"]:
            if "xl" in budgets and n <= spec.get("xl_max", 0): key, length = "xl", "xl"
            else: st.skip("too-long"); return
        k2, cell = keyer(n)
        accept(prompt, n, doc, key if length else k2, cell, think=think, has_system=has_sys(msgs), length=length)

    def do_session(doc):
        turns = doc["turns"]; tools = doc["tools"]
        flat = [m for t in turns for m in t]
        while flat and flat[0]["role"] != "user": flat = flat[1:]
        if not flat or not any(m["role"] == "assistant" for m in flat): st.skip("session-no-user-start"); return
        if has_special(json.dumps(flat, ensure_ascii=False)): st.skip("special-token"); return
        prompt, think = render(flat, tools)
        if prompt is None: return
        n = ntok(prompt)
        ex = {"turns": sum(m["role"] == "user" for m in flat)}
        if n <= XL_TOK:
            if n < spec["min_tok"]: st.skip("short"); return
            accept(prompt, n, doc, "all", spec["cell"], think=think, has_system=False, extra=ex); return
        if n <= spec["xl_max"] and used["xl"] < budgets["xl"]:
            accept(prompt, n, doc, "xl", spec["cell"], think=think, has_system=False, length="xl", extra=ex); return
        # pieces: whole turns packed up to 128k, each piece starting at a user turn
        base = ntok(render_chat([{"role": "user", "content": "x"}], tools)[0]) - 1
        pieces, cur, cur_n = [], [], base
        for t in turns:
            if t[0]["role"] != "user": st.skip("turn-not-user-start"); continue
            tp, _ = render(t, tools)
            if tp is None: continue
            tn = ntok(tp) - base
            if tn > XL_TOK - base: st.skip("turn-over-128k"); (cur and pieces.append(cur)); cur, cur_n = [], base; continue
            if cur and cur_n + tn > XL_TOK: pieces.append(cur); cur, cur_n = [], base
            cur += t; cur_n += tn
        if cur: pieces.append(cur)
        for i, p in enumerate(pieces):
            pp, th = render(p, tools)
            if pp is None: continue
            m = ntok(pp)
            if m > XL_TOK or m < spec["min_tok"]: st.skip("piece-size"); continue
            accept(pp, m, doc, "all", spec["cell"], think=th, has_system=False,
                   extra={"piece": i, "pieces": len(pieces), "orig_tokens": n, "turns": sum(x["role"] == "user" for x in p)})

    handlers = {"text": do_text, "chat": do_chat, "session": do_session}
    gen = gens.GENS[spec["gen"]](spec, rev, rng, st)
    reason = "exhausted"; others_done_at = None
    for doc in gen:
        scanned += 1
        full = lambda k: used[k] >= 0.97 * budgets[k] or k in done
        if all(full(k) for k in budgets if k != "xl"):
            if "xl" not in budgets or full("xl"): reason = "budgets-full"; break
            others_done_at = scanned if others_done_at is None else others_done_at
            if scanned - others_done_at >= 300: reason = "budgets-full (xl not reached)"; break
        if scanned >= a.max_scan: reason = "max-scan"; break
        if time.time() - t0 > a.max_seconds: reason = "max-seconds"; break
        try: handlers[doc["kind"]](doc)
        except Exception as e:
            st.skip(f"error:{type(e).__name__}")
            if st.skips[f"error:{type(e).__name__}"] <= 3: print("ERROR", sid, repr(e)[:300], file=sys.stderr, flush=True)
        if scanned % 500 == 0:
            print(f"{sid}: scanned {scanned} accepted {n_acc} tokens {st.tok} {dict(used)} {time.time() - t0:.0f}s", flush=True)
    stats = {"id": sid, "source": spec["source"], "cell": spec["cell"], "dataset": spec["repo"], "config": spec.get("config"),
             "split": spec.get("split", "train"), "revision": rev, "scale": a.scale, "budgets": budgets, "used": dict(used),
             "accepted": n_acc, "tokens": st.tok, "scanned": scanned, "stopped": reason, "files_opened": st.files,
             "skips": dict(st.skips), "dups": dict(dups), "tokens_by_band": dict(by_band), "seconds": round(time.time() - t0, 1)}
    out.close(stats)
    print("DONE", json.dumps(stats), flush=True)


def hashlib_seed(sid):
    import hashlib
    return hashlib.sha256(f"{SEED}:{sid}".encode()).hexdigest()[:12]


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)     # datasets' streaming prefetch threads can keep the process alive
