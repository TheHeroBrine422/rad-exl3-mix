#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkjudge217.py CALIB_GOLDEN CALIB_JSONL POOLS LEDGER POOLS214 ROWS CAP SEED -- #217's held-out judge slice j217: qad-pool documents of the whole-calib golden
that are in no training set of w214 / w217 (not in POOLS214's train214 pool, no `qad214` / `qad217` ledger use, so also not w214's fixed-eval documents), that the
release's own fits did not read (ledger uses swj27-qad, w94-headx-qad*, roles select / final / fit / fit-hessian) and that no other QAD task trains on
(qad215-*, qad218, qad218-tune, qad181-fixed), drawn per category in proportion to the candidates' gen rows (seeded) up to ROWS gen rows (rows scored from
the first CAP tokens of each document). Prints golden_pick.py's document arguments (idx or idx:CAP) on one line, a summary on stderr."""
import collections, json, random, sys
import numpy as np

G, CORP, POOLS, LED, POOLS214, T, CAP, SEED = sys.argv[1:9]
T, CAP, SEED = int(T), int(CAP), int(SEED)
pool = json.load(open(POOLS))["pool"]; p214 = json.load(open(POOLS214))["pool"]
USES = {"swj27-qad", "w94-headx-qad", "w94-headx-qad-127", "qad214", "qad215-train", "qad215-check", "qad217", "qad218", "qad218-tune", "qad181-fixed"}
ROLES = {"select", "final", "fit", "fit-hessian"}
use = collections.defaultdict(list)
for l in open(LED):
    r = json.loads(l)
    if r["doc"] != "all": use[r["doc"]].append((r["use"], r["role"]))
def bad(i): return p214[i] == "train214" or any(u in USES or r in ROLES or u.startswith("@reserve") for u, r in use[i])
m = json.load(open(f"{G}/kld.json")); gen = np.fromfile(f"{G}/gen.u8", np.uint8)
docs = [json.loads(l) for l in open(CORP).read().rstrip("\n").split("\n")]
assert len(docs) == len(m["docs"]) == len(pool) == len(p214)
cand = [i for i, p in enumerate(pool) if p == "qad" and not bad(i)]
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
