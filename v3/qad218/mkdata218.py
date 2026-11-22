#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkdata218.py CALIB_GOLDEN POOLS LEDGER OUT.json [--seed 218] [--cap 8192] [--judge 200000] [--tune 50000] -- #218's document split
of the calib qad pool (gen rows of the official gen.u8, each document from its start to CAP tokens):

  judge   qad218-heldout (ledger role check): qad-pool documents never read by anything that shaped the released 2.50 model
          (not in swj27-qad: the release's swj fit, pipeline/config/v3.toml [post.swj] fit_tokens 1M; not in the w94 head
          fits) nor by another
          QAD read slice (qad214, qad215-*, qad181-fixed); seeded order until >= --judge gen rows
  tune    qad218-tune (role select): the eta line search's training-side held-out set, >= --tune gen rows, from the remaining
          qad-pool documents
  train   qad218 (role train): the rest of the qad pool in a seeded order (rounds take consecutive document runs)"""
import argparse, json, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("calib"); ap.add_argument("pools"); ap.add_argument("ledger"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=218); ap.add_argument("--cap", type=int, default=8192)
ap.add_argument("--judge", type=int, default=200000); ap.add_argument("--tune", type=int, default=50000)
a = ap.parse_args()
m = json.load(open(f"{a.calib}/kld.json")); docs = m["docs"]
gen = np.memmap(f"{a.calib}/gen.u8", np.uint8, "r")
pool = json.load(open(a.pools))["pool"]
assert len(pool) == len(docs)
BLOCK = {"swj27-qad", "w94-headx-qad", "w94-headx-qad-127", "qad214", "qad215-train",
         "qad215-check", "qad181-fixed"}
used = {}
for l in open(a.ledger):
    r = json.loads(l)
    if r["doc"] != "all": used.setdefault(r["doc"], set()).add(r["use"])


def rows(i):
    d = docs[i]; n = min(a.cap, d["tokens"])
    j0, j1 = d["score_from"], n - 1
    if j1 <= j0: return 0
    r0 = d["row0"]
    return int(np.asarray(gen[r0:r0 + (j1 - j0)]).astype(bool).sum())


qad = [i for i, p in enumerate(pool) if p == "qad"]
rng = np.random.default_rng(a.seed)
free = [i for i in qad if not (used.get(i, set()) & BLOCK)]
rng.shuffle(free)
judge, jr = [], 0
for i in free:
    if jr >= a.judge: break
    n = rows(i)
    if n >= 16: judge.append(i); jr += n
assert jr >= a.judge, f"only {jr} judge rows from {len(free)} free documents"
rest = [i for i in qad if i not in set(judge)]
rng.shuffle(rest)
tune, tr = [], 0
for i in rest:
    if tr >= a.tune: break
    n = rows(i)
    if n >= 16: tune.append(i); tr += n
ts = set(tune)
train = [i for i in rest if i not in ts and rows(i) >= 16]
out = {"seed": a.seed, "cap": a.cap, "judge": judge, "judge_rows": jr, "tune": tune, "tune_rows": tr, "train": train,
       "train_rows": [rows(i) for i in train], "blocked_uses": sorted(BLOCK), "free_docs": len(free)}
json.dump(out, open(a.out, "w"))
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in out.items() if k != "blocked_uses"}),
      "train rows", sum(out["train_rows"]))
