#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cuts228.py SRC OUT [--tok DIR] [--workers N] [--seed 228] [--sessions N] -- w228 step 1+2: render every pi session whole (the
active branch, xcorp/pisess.convert + Qwen3.8's chat template with tools, preserve_thinking on as the template's default, then
pisess_redact.redact) and pick the on-policy cut points.

Render (pass 1, parallel, restartable: OUT/sess/<n>.npz per session file, skipped when it exists): the session's tokens up to the
deepest cut it can have (+ 64), the positions of every <|im_start|> (turn starts) and the candidate cuts: an assistant turn start
(`<|im_start|>assistant`) right after a user-role turn (a user message or a tool response, the template renders both as
`<|im_start|>user`), whose prefix p (tokens before that <|im_start|>) is MIN <= p <= MAX (8,192 .. 131,000: prefix + the generation
prompt + 4,096 new tokens fit the servers' 135,168). The rendered text is tokenised to ~1.2M characters at most (a session's first
~250k+ tokens); candidates closer than 64 tokens to a cut text's end are dropped.

Pick (pass 2, seeded, deterministic): a candidate is dropped when the 2,048 tokens before it were seen in an earlier-picked session
(forked / continued sessions share their history); per session <= 4 cuts at increasing depths (sessions with > 4 candidates: one at
random from each quarter of its candidates by depth); sessions in a seeded order interleaved by kind (round robin over
main / sub / worker / other, idea / miner only every 3rd round, so the ~640 one-turn mining runs do not crowd out the multi-turn
work; the generator walks this order and stops at its token target), at most
--sessions sessions. --follow (lead 11-24, option B: prefill dominates, ~30 prefill tokens a generated one): each anchor also brings the
session's next candidate (the next assistant turn after a user / tool turn), whose prompt re-prefills only the delta after the anchor's
cached prefix; a follower's key is checked against the seen set but not added, so the anchors are the same as without --follow.
Writes OUT/cuts.jsonl in generation order: {cut, sidx, file, kind, p, depth, ncand, header, follow} (header = the system turn's end;
cut = session order x 10 + depth rank, + 4 for a follower), OUT/cuts.stats.json.
PRIVATE: the user's sessions; OUT stays on our machines (box2 /data), never in the repo."""
import argparse, collections, glob, hashlib, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "xcorp"))
import numpy as np

MIN_P, MAX_P, MAX_CHARS, EDGE = 8192, 131000, 1_200_000, 64
GEN_PROMPT = "<|im_start|>assistant\n<think>\n"         # the template's add_generation_prompt with thinking on (its default)


def session_files(src):
    return sorted(glob.glob(f"{src}/--workspace--/*.jsonl") + glob.glob(f"{src}/--workspace--/*/*/run-0/session.jsonl")
                  + glob.glob(f"{src}/--tmp--/*.jsonl"))


def kind_of(rel, head):
    if rel.startswith("--workspace--/"): return "main" if rel.count("/") == 1 else "sub"
    if "local/mine/brief.md" in head or "local/mine/chunks" in head: return "miner"
    if "papers/run/instr-" in head: return "idea"
    if "local/workers/instr3/" in head: return "worker"
    return "other"


_tok = None


def tok_init(tokdir):
    global _tok
    from transformers import AutoTokenizer
    from transformers.utils import logging as hflog
    hflog.set_verbosity_error()
    _tok = AutoTokenizer.from_pretrained(tokdir)


def render_one(job):
    """one session file -> OUT/sess/<n>.npz (tokens, starts, cands) or a tiny npz with no candidates"""
    import collections as C
    import pisess, pisess_redact as red
    n, path, rel, out = job
    dst = f"{out}/sess/{n}.npz"
    if os.path.exists(dst): return n
    with open(path, errors="replace") as fh: head = fh.read(400000)
    kind = kind_of(rel, head)
    E = []
    for l in open(path, errors="replace"):
        try: E.append(json.loads(l))
        except Exception: pass

    class NoDedup:
        def __contains__(self, h): return False
        def add(self, h): pass
    pisess.SEEN_FILES = NoDedup()
    info = {"kind": kind, "file": rel, "status": "ok"}
    toks = np.zeros(0, np.int32); starts = np.zeros(0, np.int32); cands = np.zeros(0, np.int32)
    try:
        msgs, tl = pisess.convert(pisess.active_path(E), 16000)
        if not any(m["role"] == "assistant" for m in msgs): info["status"] = "no-assistant"
        else:
            if msgs[0]["role"] != "user": msgs = [{"role": "user", "content": "(continued session)"}] + msgs
            text = red.redact(_tok.apply_chat_template(msgs, tools=tl, tokenize=False), C.Counter())
            cut_text = len(text) > MAX_CHARS
            ids = np.asarray(_tok(text[:MAX_CHARS], add_special_tokens=False)["input_ids"], np.int32)
            S = _tok.convert_tokens_to_ids("<|im_start|>"); EN = _tok.convert_tokens_to_ids("<|im_end|>")
            A = _tok.encode("assistant", add_special_tokens=False)[0]; U = _tok.encode("user", add_special_tokens=False)[0]
            st = np.where(ids == S)[0]
            lim = len(ids) - EDGE if cut_text else len(ids)
            c = []
            for i in range(1, len(st)):
                p = st[i]
                if p < MIN_P or p > MAX_P or p >= lim or p + 1 >= len(ids): continue
                if ids[p + 1] != A or ids[st[i - 1] + 1] != U or ids[p - 2] != EN: continue
                c.append(p)
            info["tokens"] = int(len(ids)); info["truncated"] = bool(cut_text)
            keep = min(len(ids), (max(c) if c else MIN_P) + EDGE)
            toks = ids[:keep]; starts = st[st < keep].astype(np.int32); cands = np.asarray(c, np.int32)
    except Exception as e:
        info["status"] = f"fail {type(e).__name__}: {str(e)[:200]}"
    np.savez(dst + ".part.npz", tokens=toks, starts=starts, cands=cands, info=json.dumps(info))
    os.replace(dst + ".part.npz", dst)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--tok", default="/data/q27/models/qwen38-27b-bf16")
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--seed", type=int, default=228)
    ap.add_argument("--sessions", type=int, default=1200); ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--follow", action="store_true", help="also cut at the next assistant turn after each anchor (lead 11-24, option B)")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/sess", exist_ok=True)
    files = session_files(a.src)
    jobs = [(n, f, os.path.relpath(f, a.src), a.out) for n, f in enumerate(files)]
    json.dump([j[2] for j in jobs], open(f"{a.out}/files.json", "w"))
    todo = [j for j in jobs if not os.path.exists(f"{a.out}/sess/{j[0]}.npz")]
    print(f"{len(files)} session files, {len(todo)} to render", flush=True)
    if todo:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers, initializer=tok_init, initargs=(a.tok,)) as pool:
            for k, _ in enumerate(pool.imap_unordered(render_one, sorted(todo, key=lambda j: -os.path.getsize(j[1])), chunksize=1)):
                if (k + 1) % 200 == 0: print(f"  {k + 1} rendered", flush=True)

    # pass 2: the pick
    S = {}
    st = collections.Counter()
    for n, f, rel, _ in jobs:
        z = np.load(f"{a.out}/sess/{n}.npz"); info = json.loads(str(z["info"]))
        st[f"status_{info['status'].split(' ')[0]}"] += 1
        if len(z["cands"]): S[n] = (info, z["cands"].tolist())
    st["sessions_with_cuts"] = len(S); st["candidates"] = sum(len(c) for _, c in S.values())
    rng = random.Random(a.seed)
    by_kind = collections.defaultdict(list)
    for n in sorted(S): by_kind[S[n][0]["kind"]].append(n)
    for k in by_kind: rng.shuffle(by_kind[k])
    order, kinds = [], ["main", "sub", "worker", "other", "idea", "miner"]
    ptr = {k: 0 for k in kinds}; rnd = 0
    while any(ptr[k] < len(by_kind[k]) for k in kinds):
        for k in kinds:
            if k in ("idea", "miner") and rnd % 3: continue        # one-turn print-mode runs: every 3rd round only
            if ptr[k] < len(by_kind[k]): order.append(by_kind[k][ptr[k]]); ptr[k] += 1
        rnd += 1
    seen, rows, ns = set(), [], 0
    for n in order:
        if ns >= a.sessions: break
        info, c = S[n]
        z = np.load(f"{a.out}/sess/{n}.npz"); t = z["tokens"]; starts = z["starts"]
        keyed = []
        for p in c:
            h = hashlib.sha1(t[p - 2048:p].tobytes()).hexdigest()
            if h not in seen: keyed.append((p, h))
        if not keyed: st["sessions_all_dup"] += 1; continue
        if len(keyed) > a.per:
            q = np.array_split(np.arange(len(keyed)), a.per)
            keyed = [keyed[rng.choice(list(b))] for b in q]
        for p, h in keyed: seen.add(h)
        header = int(starts[1]) if len(starts) > 1 else 0          # the system turn (tools block) ends where turn 2 starts
        picked = {p for p, _ in keyed}
        for r, (p, h) in enumerate(keyed):
            rows.append({"cut": ns * 10 + r, "sidx": n, "file": info["file"], "kind": info["kind"], "p": int(p), "depth": r,
                         "ncand": len(c), "header": header, "follow": False})
            i = c.index(p)                                           # --follow: the session's next assistant turn too
            if a.follow and i + 1 < len(c) and c[i + 1] not in picked:
                f = c[i + 1]
                if hashlib.sha1(t[f - 2048:f].tobytes()).hexdigest() in seen: continue   # checked, not added: anchors stay as without --follow
                picked.add(f)
                rows.append({"cut": ns * 10 + 4 + r, "sidx": n, "file": info["file"], "kind": info["kind"], "p": int(f), "depth": r,
                             "ncand": len(c), "header": header, "follow": True})
        ns += 1
    st["sessions_picked"] = ns; st["cuts"] = len(rows)
    st["cuts_by_kind"] = dict(collections.Counter(r["kind"] for r in rows))
    st["sessions_by_kind"] = dict(collections.Counter(S[n][0]["kind"] for n in {r["sidx"] for r in rows}))
    st["candidates_by_kind"] = dict(collections.Counter(S[n][0]["kind"] for n in S for _ in S[n][1]))
    st["depth_hist_k"] = dict(sorted(collections.Counter(min(r["p"] // 16384 * 16, 128) for r in rows).items()))
    with open(f"{a.out}/cuts.jsonl.part", "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    os.replace(f"{a.out}/cuts.jsonl.part", f"{a.out}/cuts.jsonl")
    json.dump(st, open(f"{a.out}/cuts.stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
