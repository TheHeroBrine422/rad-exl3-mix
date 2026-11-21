#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkjudge214.py CALIB_GOLDEN CALIB_JSONL POOLS LEDGER ROWS CAP SEED -- #214's judge slice: verify-pool documents of the whole-calib golden
that the ledger shows no selection / final / fit use of, never in the V4 reserve, in no QAD judge slice before (golden-verify27-w114*) and
in no QAD training use, drawn per category in proportion to the candidates' gen rows (seeded) up to ROWS gen rows (rows scored from the
first CAP tokens of each document). Prints golden_pick.py's document arguments (idx or idx:CAP) on one line, a summary on stderr."""
import collections, json, random, sys
import numpy as np

G, CORP, POOLS, LED, T, CAP, SEED = sys.argv[1:8]
T, CAP, SEED = int(T), int(CAP), int(SEED)
pool = json.load(open(POOLS))["pool"]
use = collections.defaultdict(list)
for l in open(LED):
    r = json.loads(l)
    if r["doc"] != "all": use[r["doc"]].append((r["use"], r["role"]))
def bad(i):
    for u, r in use[i]:
        if r in ("select", "final", "fit-hessian", "fit-diag", "fit") or u.startswith("@reserve") or "verify27-w114" in u or "qad" in u: return True
    return False
m = json.load(open(f"{G}/kld.json")); gen = np.fromfile(f"{G}/gen.u8", np.uint8)
docs = [json.loads(l) for l in open(CORP).read().rstrip("\n").split("\n")]
assert len(docs) == len(m["docs"]) == len(pool)
cand = [i for i, p in enumerate(pool) if p == "verify" and not bad(i)]
def rows(i):
    d = m["docs"][i]; n = min(d["tokens"], CAP)
    a, b = d["row0"], d["row0"] + max(0, n - 1 - d["score_from"])
    return int(gen[a:b].sum())
bycat = collections.defaultdict(list)
for i in cand: bycat[docs[i]["category"]].append(i)
tot = {c: sum(rows(i) for i in v) for c, v in bycat.items()}; allr = sum(tot.values())
rng = random.Random(SEED); out = []; got = collections.Counter()
for c in sorted(bycat):
    want = T * tot[c] / allr; idx = bycat[c][:]; rng.shuffle(idx)
    for i in idx:
        if got[c] >= want: break
        if rows(i) < 16: continue
        got[c] += rows(i); out.append(f"{i}:{CAP}" if m["docs"][i]["tokens"] > CAP else str(i))
print(" ".join(sorted(out, key=lambda s: int(s.split(":")[0]))))
print(f"{len(cand)} candidates ({allr} gen rows at cap {CAP}); picked {len(out)} documents, {sum(got.values())} gen rows: " + ", ".join(f"{c} {got[c]}" for c in sorted(got)), file=sys.stderr)
