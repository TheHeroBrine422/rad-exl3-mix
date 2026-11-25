#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cuts230.py XGAP_DIR TOKENIZER OUT [--seed 230] [--t1-cuts 1600] [--t2-cuts 2400] [--t1-judge 130] [--t2-judge 200] -- w230 step 1:
the cut lists of the two teacher sets (public sources only: xgap-v1 / xcorp-v1 docs, never the user's sessions), built from the
rendered Qwen3.8 documents of XGAP_DIR/manifest.jsonl (+ corpus/). Output: OUT/{t1,t1-judge,t2,t2-judge}/cuts.jsonl + sess/<sidx>.npz
(tokens of the prompt prefix, the same layout gen230.py / docs230.py read) and OUT/stats.json, OUT/samples.txt (3 rendered prompts a set).

Set 1 (agent cuts, WHOLE prefixes): docs = xgap-v1 cell agent-sessions + xcorp-v1 category agentic. A cut = the start of an assistant
turn (<|im_start|>assistant\\n) that follows a user / tool turn (tool results render as user turns), prefix p = the tokens before it,
4,096 <= p <= 96,000 (lead 11-25: the deepest band costs ~40 % of prefill). Generation header = <|im_start|>assistant\\n<think>\\n when
the doc's system turn carries the "Reasoning effort is set to ..." line, else the thinking-off form <think>\\n\\n</think>\\n\\n.
Sessions: an xgap session = (src_id, session) and may hold several docs (chunks); an xcorp session = the doc. A session with ANY
held-out doc goes wholly to the judge side. Judge sessions: all xgap holdout sessions + a seeded 10 % of the xcorp agentic docs.
Depth mix by cut count 40 / 30 / 20 / 10 % over 4-16k / 16-32k / 32-64k / 64-96k, <= 4 anchors a session (one a band) plus the NEXT
candidate of the same doc after each anchor (a follower; counted in its own band); bands are filled scarcest first, sessions in a
seeded order. Judge: one anchor + follower a session. The cuts' session order = the queue order (seeded shuffle).
Set 2 (chat at varied settings): one cut a doc, a random assistant turn after a user turn with prefix <= 32,768 tokens, history as
rendered. Groups (weight): xgap multilingual-chat (2), long-chat (1), tools-other (1), xcorp chat / tools / reasoning (1 each). Slots
go by smooth weighted round robin over the groups; setting = slot index mod 4 over xhigh, low, medium, off. The system turn is edited
as the template would: xhigh / low carry the effort line first (a system turn made if there was none), medium / off carry none (a
system turn that held only the line is dropped), real system prompts stay; off uses the thinking-off generation header.
Judge pools: xgap holdout docs + a seeded 10 % of the xcorp docs (by doc); training excludes both."""
import argparse, collections, json, os, random, sys

import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("xgap"); ap.add_argument("tok"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=230)
ap.add_argument("--t1-cuts", type=int, default=1600); ap.add_argument("--t2-cuts", type=int, default=2400)
ap.add_argument("--t1-judge", type=int, default=130); ap.add_argument("--t2-judge", type=int, default=200)
a = ap.parse_args()
tk = AutoTokenizer.from_pretrained(a.tok)
IM_START, IM_END, NL = 248045, 248046, 198
ASSIST = tk.encode("assistant", add_special_tokens=False); USER = tk.encode("user", add_special_tokens=False)
SYSTEM = tk.encode("system", add_special_tokens=False)
assert len(ASSIST) == len(USER) == len(SYSTEM) == 1 and tk.encode("<|im_start|>", add_special_tokens=False) == [IM_START]
ASSIST, USER, SYSTEM = ASSIST[0], USER[0], SYSTEM[0]
GP_ON = tk.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
GP_OFF = tk.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
assert GP_ON == [248045, 74455, 198, 248068, 198], GP_ON
EFFORT = {"xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
                   "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
          "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
                 "unnecessary elaboration."}   # chat_template.jinja lines 45-60 (medium: no line)
SETTINGS = ["xhigh", "low", "medium", "off"]
BANDS = [(4096, 16384), (16384, 32768), (32768, 65536), (65536, 96001)]; MIX = [0.4, 0.3, 0.2, 0.1]
P_MAX1 = 96000; P_MAX2 = 32768; MAXNEW = 32768; MAXNEW_OFF = 16384

man = [json.loads(l) for l in open(f"{a.xgap}/manifest.jsonl")]
_files = {}


def doc_text(m):
    if m["file"] not in _files: _files[m["file"]] = open(f"{a.xgap}/{m['file']}").read().rstrip("\n").split("\n")
    return json.loads(_files[m["file"]][m["line"]])["prompt"]


def enc(text): return tk(text, add_special_tokens=False)["input_ids"]


def candidates(ids):
    """prefix lengths i (the index of an <|im_start|>) of assistant turns that follow a user turn"""
    st = [i for i, t in enumerate(ids) if t == IM_START]
    out = []
    for k in range(1, len(st)):
        i = st[k]
        if ids[i + 1] == ASSIST and ids[i + 2] == NL and ids[st[k - 1] + 1] == USER: out.append(i)
    return out


def thinking_on(text): return text.startswith("<|im_start|>system\nReasoning effort is set to")


def band(p):
    for b, (lo, hi) in enumerate(BANDS):
        if lo <= p < hi: return b
    return None


def write_set(name, cuts, sess, extra=None):
    d = f"{a.out}/{name}"; os.makedirs(f"{d}/sess", exist_ok=True)
    for sidx, toks in sess.items(): np.savez(f"{d}/sess/{sidx}.npz", tokens=np.asarray(toks, np.int32))
    with open(f"{d}/cuts.jsonl", "w") as f:
        for c in cuts: f.write(json.dumps(c) + "\n")


# ---------------------------------------------------------------------------------------------------------------- set 1
def set1():
    docs = [m for m in man if (m["origin"] == "xgap-v1" and m["cell"] == "agent-sessions") or (m["origin"] == "xcorp-v1" and m["category"] == "agentic")]
    rng = random.Random(f"{a.seed}-t1")
    xc = sorted(m["id"] for m in docs if m["origin"] == "xcorp-v1"); rng.shuffle(xc); xc_hold = set(xc[:round(0.1 * len(xc))])
    def skey(m): return f"xg:{m['src_id']}:{m['session']}" if m["origin"] == "xgap-v1" and m.get("session") else f"doc:{m['id']}"
    units = collections.OrderedDict()
    for m in docs: units.setdefault(skey(m), []).append(m)
    hold_units = {k for k, ms in units.items() if any(m.get("holdout") or m["id"] in xc_hold for m in ms)}
    print(f"set1: {len(docs)} docs, {len(units)} sessions, judge sessions {len(hold_units)}", flush=True)
    info = {}; mism = collections.Counter()
    for k, ms in units.items():
        for m in ms:
            text = doc_text(m); ids = enc(text); on = thinking_on(text)
            mism[(m.get("thinking"), on)] += 1
            cs = [p for p in candidates(ids) if 4096 <= p <= P_MAX1]
            info[m["id"]] = dict(ids=ids, on=on, cands=cs, m=m)
    print("set1 manifest thinking vs rendered effort line:", dict(mism), flush=True)

    def pick(unit_keys, max_anchors, n_target, tag):
        r = random.Random(f"{a.seed}-t1-{tag}"); order = sorted(unit_keys); r.shuffle(order)
        quota = [round(n_target * w) for w in MIX]; got = [0] * 4; chosen = collections.defaultdict(list)   # unit -> [(doc id, p, follower)]
        usedb = collections.defaultdict(set)
        for b in [3, 2, 1, 0]:                                                              # scarcest band first
            for k in order:
                if got[b] >= quota[b]: break
                if len(chosen[k]) and sum(1 for x in chosen[k] if not x[2]) >= max_anchors: continue
                if b in usedb[k]: continue
                opts = [(m["id"], p) for m in units[k] for p in info[m["id"]]["cands"] if band(p) == b and not any(x[0] == m["id"] and x[1] == p for x in chosen[k])]
                if not opts: continue
                did, p = r.choice(opts); chosen[k].append((did, p, False)); usedb[k].add(b); got[b] += 1
                cs = info[did]["cands"]; j = cs.index(p)
                if j + 1 < len(cs) and not any(x[0] == did and x[1] == cs[j + 1] for x in chosen[k]):
                    chosen[k].append((did, cs[j + 1], True)); got[band(cs[j + 1])] += 1
        return order, chosen, quota, got

    train_units = [k for k in units if k not in hold_units]
    out = {}
    for name, keys, ma, n, tag in (("t1", train_units, 4, a.t1_cuts, "train"), ("t1-judge", sorted(hold_units), 1, a.t1_judge, "judge")):
        order, chosen, quota, got = pick(keys, ma, n, tag)
        cuts = []
        for gi, k in enumerate(o for o in order if chosen[o]):
            for did, p, fol in sorted(chosen[k], key=lambda x: (x[0], x[1])):
                inf = info[did]; m = inf["m"]
                cuts.append({"cut": len(cuts), "sidx": did, "group": gi, "unit": k, "p": p, "depth": p, "band": band(p), "follower": fol,
                             "kind": m["source"], "origin": m["origin"], "doc": m["id"], "setting": "xhigh_on" if inf["on"] else "off",
                             "gp": GP_ON if inf["on"] else GP_OFF, "max_new": MAXNEW if inf["on"] else MAXNEW_OFF,
                             "holdout": name.endswith("judge")})
        # trim each doc's tokens to its deepest cut; renumber cuts in the queue order (sessions as ordered, cuts by depth inside)
        deepest = collections.defaultdict(int)
        for c in cuts: deepest[c["sidx"]] = max(deepest[c["sidx"]], c["p"])
        sess = {s: info[s]["ids"][:deepest[s]] for s in deepest}
        cuts.sort(key=lambda c: (c["group"], c["p"]))
        for i, c in enumerate(cuts): c["cut"] = i
        write_set(name, cuts, sess)
        out[name] = {"cuts": len(cuts), "sessions": len({c["unit"] for c in cuts}), "docs": len(sess), "quota": quota, "got_by_band": [sum(1 for c in cuts if c["band"] == b) for b in range(4)],
                     "followers": sum(c["follower"] for c in cuts), "thinking_on": sum(c["setting"] != "off" for c in cuts),
                     "prefill_tokens": sum(c["p"] for c in cuts), "by_origin": dict(collections.Counter(c["origin"] for c in cuts)),
                     "by_source": dict(collections.Counter(c["kind"] for c in cuts)),
                     "cuts_per_session": dict(collections.Counter(collections.Counter(c["unit"] for c in cuts).values()))}
        print(name, json.dumps(out[name]), flush=True)
    out["t1_thinking_check"] = {str(k): v for k, v in mism.items()}
    return out


# ---------------------------------------------------------------------------------------------------------------- set 2
def edit_head(text, setting):
    """(new head text, length of the original system turn incl. its <|im_end|>\\n) for a doc whose rendering starts with the template's system turn"""
    n0 = 0; sysblock = ""
    if text.startswith("<|im_start|>system\n"):
        e = text.index("<|im_end|>\n"); sysblock = text[len("<|im_start|>system\n"):e]; n0 = e + len("<|im_end|>\n")
    rest = sysblock
    for line in EFFORT.values():
        if rest.startswith(line):
            rest = rest[len(line):]
            if rest.startswith("\n\n"): rest = rest[2:]
            else: assert rest == "", rest[:80]
            break
    line = EFFORT[setting] if setting in EFFORT else ""
    body = line + ("\n\n" if line and rest else "") + rest
    return ("<|im_start|>system\n" + body + "<|im_end|>\n" if body else ""), n0


def set2():
    GROUPS = {"xg-multilingual-chat": 2, "xg-long-chat": 1, "xg-tools-other": 1, "xc-chat": 1, "xc-tools": 1, "xc-reasoning": 1}
    def grp(m):
        if m["kind"] != "chat": return None
        if m["origin"] == "xgap-v1": return {"multilingual-chat": "xg-multilingual-chat", "long-chat": "xg-long-chat", "tools-other": "xg-tools-other"}.get(m["cell"])
        if m["origin"] == "xcorp-v1": return {"chat": "xc-chat", "tools": "xc-tools", "reasoning": "xc-reasoning"}.get(m["category"])
    rng = random.Random(f"{a.seed}-t2")
    pool = collections.defaultdict(lambda: ([], []))   # group -> (train docs, judge docs)
    for m in man:
        g = grp(m)
        if g: pool[g][1 if m.get("holdout") else 0].append(m)
    for g in sorted(pool):
        tr = sorted(pool[g][0], key=lambda m: m["id"]); rng.shuffle(tr)
        if g.startswith("xc-"):
            h = round(0.1 * len(tr)); pool[g] = (tr[h:], tr[:h])
        else: pool[g] = (tr, sorted(pool[g][1], key=lambda m: m["id"]))
        rng.shuffle(pool[g][1])
        print(f"set2 {g}: train pool {len(pool[g][0])} judge pool {len(pool[g][1])}", flush=True)
    out = {}; samples = []
    for name, side, n in (("t2", 0, a.t2_cuts), ("t2-judge", 1, a.t2_judge)):
        r = random.Random(f"{a.seed}-t2-{name}"); cur = {g: 0 for g in GROUPS}; tot = sum(GROUPS.values()); nxt = {g: 0 for g in GROUPS}
        cuts, sess = [], {}; skipped = collections.Counter(); checked = 0
        for i in range(n):
            for g in GROUPS: cur[g] += GROUPS[g]
            g = max(cur, key=cur.get); cur[g] -= tot; setting = SETTINGS[i % 4]
            while True:
                docs = pool[g][side]
                if nxt[g] >= len(docs): skipped[f"{g}:exhausted"] += 1; m = None; break
                m = docs[nxt[g]]; nxt[g] += 1
                text = doc_text(m); ids = enc(text)
                head, n0 = edit_head(text, setting)
                n0t = 0
                if n0:
                    n0t = ids.index(IM_END) + 2; assert ids[n0t - 1] == NL
                new = (enc(head) if head else []) + ids[n0t:]
                if checked < 60:
                    full = enc(head + text[n0:]); assert full == new, (m["id"], setting); checked += 1
                cs = [p for p in candidates(new) if p <= P_MAX2]
                if not cs: skipped[f"{g}:no_cut_le_32k"] += 1; continue
                p = r.choice(cs); break
            if m is None: continue
            on = setting != "off"; sidx = len(cuts)
            sess[sidx] = new[:p]
            cuts.append({"cut": sidx, "sidx": sidx, "group": sidx, "p": p, "depth": p, "band": None, "follower": False, "kind": g, "origin": m["origin"],
                         "doc": m["id"], "source": m["source"], "setting": setting, "gp": GP_ON if on else GP_OFF,
                         "max_new": MAXNEW if on else MAXNEW_OFF, "holdout": side == 1, "doc_thinking": m.get("thinking")})
        write_set(name, cuts, sess)
        out[name] = {"cuts": len(cuts), "by_group": dict(collections.Counter(c["kind"] for c in cuts)), "by_setting": dict(collections.Counter(c["setting"] for c in cuts)),
                     "by_origin": dict(collections.Counter(c["origin"] for c in cuts)), "skipped": dict(skipped), "prefill_tokens": sum(c["p"] for c in cuts),
                     "depth_hist": {k: sum(1 for c in cuts if lo <= c["p"] < hi) for k, (lo, hi) in {"<1k": (0, 1024), "1-4k": (1024, 4096), "4-16k": (4096, 16384), "16-32k": (16384, 32769)}.items()},
                     "group_x_setting": {f"{g}|{s}": sum(1 for c in cuts if c["kind"] == g and c["setting"] == s) for g in GROUPS for s in SETTINGS}}
        print(name, json.dumps(out[name]), flush=True)
    return out


os.makedirs(a.out, exist_ok=True)
st = {"set1": set1(), "set2": set2()}
json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1)
with open(f"{a.out}/samples.txt", "w") as f:
    for name in ("t1", "t1-judge", "t2", "t2-judge"):
        cuts = [json.loads(l) for l in open(f"{a.out}/{name}/cuts.jsonl")]
        r = random.Random(f"{a.seed}-sample-{name}")
        pick = r.sample(range(len(cuts)), 3) if name in ("t1", "t2") else r.sample(range(len(cuts)), 1)
        if name == "t2":
            for s in SETTINGS:
                j = next((i for i, c in enumerate(cuts) if c["setting"] == s and i not in pick), None)
                if j is not None: pick.append(j)
        for i in pick:
            c = cuts[i]; toks = np.load(f"{a.out}/{name}/sess/{c['sidx']}.npz")["tokens"][:c["p"]].tolist() + c["gp"]
            t = tk.decode(toks, skip_special_tokens=False)
            f.write(f"\n===== {name} cut {c['cut']} {c['kind']} setting {c['setting']} p {c['p']} max_new {c['max_new']} doc {c['doc']}\n--- HEAD\n{t[:600]}\n--- TAIL\n{t[-400:]}\n")
print("done")
