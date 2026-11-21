#!/usr/bin/env python3
"""mkpools41.py MAIN100B.jsonl LONG1M.jsonl OUT.json -- DS4.1-F pools per eval/ds41/RULES-pools.md (run once; read-only after).

Group = session or name; per category, groups in sha256("ds41pools-v1:" + group) order, cut by cumulative tokens: holdout ~2 %
(h1 / h2 alternating), then of the rest fit+search 40 %, qad 25 %, verify 15 %, reserve 20 %. g4: 4 long-verify, 4 reserve.
"""
import hashlib, json, sys, collections

main, long1m, out = sys.argv[1:4]
H = lambda g: hashlib.sha256(("ds41pools-v1:" + g).encode()).hexdigest()
docs = []
for l in open(main):
    r = json.loads(l)
    docs.append((r["name"], r.get("session") or r["name"], r["category"], r["tokens"]))
groups = collections.defaultdict(lambda: [0, []])          # (cat, group) -> [tokens, names]
for name, g, cat, tok in docs:
    groups[(cat, g)][0] += tok; groups[(cat, g)][1].append(name)
bycat = collections.defaultdict(list)
for (cat, g), (tok, names) in groups.items():
    bycat[cat].append((H(g), g, tok, names))
pool = {}
summary = collections.defaultdict(lambda: collections.Counter())
for cat, gs in sorted(bycat.items()):
    gs.sort()
    total = sum(t for _, _, t, _ in gs)
    hold_cut = 0.02 * total
    acc = 0; hold = []; rest = []
    for item in gs:
        (hold if acc < hold_cut else rest).append(item)
        acc += item[2]
    for i, (_, g, t, names) in enumerate(hold):
        p = "h1" if i % 2 == 0 else "h2"
        for n in names: pool[n] = p
        summary[cat][p] += t
    rt = sum(t for _, _, t, _ in rest); acc = 0
    for _, g, t, names in rest:
        f = acc / rt if rt else 0
        p = "fit" if f < 0.40 else "qad" if f < 0.65 else "verify" if f < 0.80 else "reserve"
        for n in names: pool[n] = p
        summary[cat][p] += t; acc += t
longs = []
for l in open(long1m):
    r = json.loads(l); longs.append((H(r.get("session") or r["name"]), r["name"], r["tokens"]))
longs.sort()
lpool = {n: ("long-verify" if i < 4 else "reserve") for i, (_, n, _) in enumerate(longs)}
tot = collections.Counter()
for c in summary.values(): tot.update(c)
json.dump({"rules": "eval/ds41/RULES-pools.md", "version": "ds41pools-v1", "pool": pool, "long1m": lpool,
           "tokens_by_category": {k: dict(v) for k, v in summary.items()}, "tokens_total": dict(tot)}, open(out, "w"))
print(json.dumps(dict(tot)), "long1m:", lpool)
