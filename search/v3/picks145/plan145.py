#!/usr/bin/env python3
"""plan145.py LEDGER POOLS VERFILE CALTOK OUT.json [--c3 POSITIONS] [--cap 8192] [--seed 145] -- the final-pick slice's
reused-verify part (c3): which verify documents, from the ledger only (no golden is read; w145, RULES-picks "Final picks").

Verify short documents (< 100k tokens; the four verify traces are handled whole as units elsewhere) are classified by
their ledger uses:
  ver4      golden-ver4 (the c1 part itself) -- not c3
  reserve0  the untouched V4 reserve: every use is @reserve-v4 (or 0.8B proxy checks) -- kept unspent, not c3
  ver3      any golden-ver3* / ref134 use (the #171 monitor, ~100 scorings; spent for selection) -- excluded
  headx     w119-headx-verify (read after the head-layout decisions, the candidates carry those heads) -- excluded
  reserve   other V4-reserve documents (already touched by V2-era or 27B checks): left to V4 -- excluded
  fitted    any fit / train role (@yaqa H_out, @headsens head diagonals: V3 inputs) -- excluded
  reuse     the rest: V1 / V2-era checks (other models), 0.8B proxy checks, golden-verify27(-w114) QAD checks
            (#165: no mix selection rode on them) -- c3 is drawn from here, fewest 27B-era (v3 / v3.1) uses first,
            then seeded random, until --c3 scored positions at --cap tokens a document.
Prints the classes and the draw; OUT.json = {"classes": {...}, "c3": [doc ids], "c3_positions", ...}."""
import argparse, collections, json, random

ap = argparse.ArgumentParser()
ap.add_argument("ledger"); ap.add_argument("pools"); ap.add_argument("verfile"); ap.add_argument("caltok"); ap.add_argument("out")
ap.add_argument("--c3", type=int, default=150000); ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--seed", type=int, default=145)
a = ap.parse_args()
pool = json.load(open(a.pools))["pool"]; tok = json.load(open(a.caltok)); V = json.load(open(a.verfile))
ver4 = set(V["ver4"]["docs"]) | {V["ver4"]["long_trace"]["doc"]}
uses = collections.defaultdict(list)
for l in open(a.ledger):
    r = json.loads(l)
    if r["doc"] != "all": uses[int(r["doc"])].append(r)
PROXY = ("proxy-08", "proxy-2b")
cls = {}
for i, p in enumerate(pool):
    if p != "verify" or tok[i] >= 100000: continue
    u = uses[i]; names = {r["use"] for r in u}
    if i in ver4: c = "ver4"
    elif any(n.startswith("golden-ver3") or n.startswith("ref134") for n in names): c = "ver3"
    elif "w119-headx-verify" in names: c = "headx"
    elif all(r["use"] == "@reserve-v4" or r["line"] in PROXY for r in u): c = "reserve0"
    elif "@reserve-v4" in names: c = "reserve"
    elif any(r["role"].startswith(("fit", "train")) for r in u): c = "fitted"
    else: c = "reuse"
    cls[i] = c
pos = lambda i: min(tok[i], a.cap) - 1
summary = {c: {"docs": sum(1 for x in cls.values() if x == c), "positions_at_cap": sum(pos(i) for i, x in cls.items() if x == c)} for c in sorted(set(cls.values()))}
cand = [i for i, c in cls.items() if c == "reuse"]
n27 = {i: sum(1 for r in uses[i] if r["line"] in ("v3", "v3.1")) for i in cand}
rng = random.Random(a.seed); cand.sort(key=lambda i: (n27[i], rng.random()))
pick, tot = [], 0
for i in cand:
    if tot >= a.c3: break
    pick.append(i); tot += pos(i)
pick.sort()
used_by = collections.Counter(r["use"] for i in pick for r in uses[i])
out = {"cap": a.cap, "seed": a.seed, "target": a.c3, "classes": summary, "c3": pick, "c3_docs": len(pick), "c3_positions": tot,
       "c3_27b_uses": dict(collections.Counter(n27[i] for i in pick)), "c3_prior_uses": dict(used_by.most_common()),
       "c3_max_doc_positions": max(pos(i) for i in pick) if pick else 0}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "c3"}, indent=1))
