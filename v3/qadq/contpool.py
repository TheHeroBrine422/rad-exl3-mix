#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""contpool.py CALIB_GOLDEN POOLS_V3 RUNDIR EXT_PLAN J217_KLD JLONG_IDS OUT_DIR [--judge-rows 1e6] [--cap 8192] [--seed 0]
-- lead 11-24 (user): the calib documents a continuation of a finished run may train on, split into a held-out judge part and a
training part. Eligible: pools-v3 qad / fit / search (V3.1 data rule: verify stays out), minus the run's own calib documents (its
base items RUNDIR/items.json + the plan-ext calib items + its fixed-eval documents), minus j217's documents, minus golden-jlong's 74.
Rows are counted as the trainer counts them (data181.Gold.nrows at CAP, documents with >= 16 rows). The judge part is drawn per
source category in proportion to the candidates' rows (seeded) up to --judge-rows; the rest is the training part.
Writes OUT_DIR/contpool.json: {"judge": [doc ids], "train": [[doc, rows], ...], summary}."""
import collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D

a = sys.argv[1:]
G, POOLS, RUN, EXT, J217, JL, OUT = a[:7]
opt = lambda k, d: type(d)(float(a[a.index(k) + 1])) if k in a else d
JR, CAP, SEED = opt("--judge-rows", 1e6), opt("--cap", 8192), opt("--seed", 0)
g = D.Gold(G, genfile=os.path.join(G, "gen.u8"))
pool = json.load(open(POOLS))["pool"]
it = json.load(open(os.path.join(RUN, "items.json")))
key = json.loads(it["key"]); ci = key["chunks"].index(next(c for c in key["chunks"] if c.rstrip("/").endswith("calib")))
used = {x[1] for x in it["items"] if x[0] == ci}
used |= {x[1] for x in json.load(open(EXT))["items"] if x[0] == "calib"}
used |= set(json.load(open(os.path.join(RUN, "fixed-eval-docs.json"))))
j217 = {d["orig"] for d in json.load(open(J217))["docs"]}
jl = set(json.load(open(JL)))
src = [d["source"] for d in json.load(open(os.path.join(G, "kld.json")))["docs"]]
cand, why = [], collections.Counter()
for i, p in enumerate(pool):
    if p not in ("qad", "fit", "search"): why["verify"] += 1; continue
    if i in used: why["used by the run"] += 1; continue
    if i in j217: why["j217"] += 1; continue
    if i in jl: why["golden-jlong"] += 1; continue
    r = g.nrows(i, CAP)
    if r < 16: why["< 16 rows"] += 1; continue
    cand.append((i, r))
cat = lambda i: src[i].split("/")[0]
bycat = collections.defaultdict(list)
for i, r in cand: bycat[cat(i)].append((i, r))
tot = sum(r for _, r in cand); rng = np.random.default_rng(SEED); judge = set()
for c, xs in sorted(bycat.items()):
    want = JR * sum(r for _, r in xs) / tot; got = 0
    for k in rng.permutation(len(xs)):
        if got >= want: break
        judge.add(xs[k][0]); got += xs[k][1]
train = [[i, r] for i, r in cand if i not in judge]
s = {"eligible_docs": len(cand), "eligible_rows": tot, "excluded": dict(why), "used_by_run": len(used),
     "judge_docs": len(judge), "judge_rows": sum(r for i, r in cand if i in judge),
     "train_docs": len(train), "train_rows": sum(r for _, r in train), "cap": CAP, "seed": SEED,
     "by_pool_train_rows": dict(collections.Counter({p: 0 for p in ("qad", "fit", "search")}) + collections.Counter({pool[i]: r for i, r in train if False}))}
bp = collections.Counter()
for i, r in train: bp[pool[i]] += r
s["by_pool_train_rows"] = dict(bp)
os.makedirs(OUT, exist_ok=True)
json.dump({"judge": sorted(judge), "train": train, "summary": s}, open(os.path.join(OUT, "contpool.json"), "w"))
print(json.dumps(s, indent=1))
