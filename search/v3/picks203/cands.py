# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cands.py DRV EVALS OUT.json [--final] -- phase-1 candidate set (RULES-picks amendment 2026-11-16 §1).
Per size: every stage-2 accept / join whose tag index >= 1000; the latest s186e re-selection's top 3; the live incumbent
and pool members; at 6.5 / 7.0 stage-1 tentatives whose stage-2 |z| <= 0.5. Dedup by assignment hash (moves.ahash).
Each candidate carries its exact eval mix from the coordinator's evals.jsonl (same base, levels, donors, act).
Without --final the live incumbent / pool are the current ones (a preview); --final after the search stops."""
import hashlib, json, os, sys

drv, evals, out = sys.argv[1:4]
st = [json.loads(l) for l in open(f"{drv}/state.jsonl")]
mix = {}
for l in open(evals):
    e = json.loads(l); mix[e["tag"]] = e["mix"]
idx = lambda t: int(t.rsplit("-", 1)[1]) if t.rsplit("-", 1)[1].isdigit() else -1
C, why = {}, {}
def add(size, tag, w):
    C.setdefault(size, []); why.setdefault(tag, [])
    if tag not in C[size]: C[size].append(tag)
    why[tag].append(w)
inc, pool, resel = {}, {}, {}
for e in st:
    ev = e.get("ev")
    if ev == "decision" and e.get("stage") == 2 and e.get("decision") in ("accept", "join") and idx(e["tag"]) >= 1000:
        add(float(e["size"]), e["tag"], "confirmed")
    if ev == "decision" and e.get("stage") == 2 and e.get("decision") == "reject" and float(e["size"]) in (6.5, 7.0) and abs(e.get("z", 9)) <= 0.5:
        add(float(e["size"]), e["tag"], "narrow-miss")
    if ev == "incumbent": inc[float(e["size"])] = e["tag"]
    if ev == "pool": pool[float(e["size"])] = e.get("members") or [e.get("tag")]
    if ev == "reselect" and e.get("slice") == "s186e": resel[float(e["size"])] = [r[0] for r in e.get("ranking", [])[:3]]
for s, t in inc.items(): add(s, t, "incumbent")
for s, m in pool.items():
    for t in m:
        if t: add(s, t, "pool")
for s, m in resel.items():
    for t in m: add(s, t, "reselect-s186e")
res = {}
for s in sorted(C):
    if s == 1.25: continue
    seen = {}
    for t in C[s]:
        p = f"{drv}/assign/{t}.json"
        if not os.path.exists(p) or t not in mix: print("missing", t, file=sys.stderr); continue
        h = hashlib.sha1(json.dumps(json.load(open(p)), sort_keys=True).encode()).hexdigest()[:16]
        if h in seen: seen[h]["why"] += why[t]; seen[h]["dups"].append(t); continue
        seen[h] = {"tag": t, "hash": h, "why": list(why[t]), "dups": [], "mix": mix[t]}
    res[str(s)] = list(seen.values())
json.dump(res, open(out, "w"))
print({s: len(v) for s, v in res.items()}, "total", sum(len(v) for v in res.values()))
