#!/usr/bin/env python3
"""calib_slice.py CORPUS.jsonl TOKENS CAP SEED [HALF] -- pick a slice of the calib corpus for fast evals: documents drawn at
random (fixed SEED) within each category, each category's share of TOKENS its share of the corpus's tokens,
every document cut to CAP tokens. HALF 0/1: only documents with index % 2 == HALF (calib half A = 0 for
proxy calibration, B = 1 for verifying mixes). HALF fit/search/qad/verify: only documents of that pool in
$RXM_WORK/challenge/pools.json (POOLS env overrides). Prints golden_pick.py's document arguments (idx or idx:cap), one line."""
import os
import json, random, sys, collections
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
corp, T, cap, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
half = sys.argv[5] if len(sys.argv) > 5 else "-1"
docs = [json.loads(l) for l in open(corp)]
# EXCLUDE_GOLDEN=dir1,dir2: skip documents already in those goldens (their kld.json docs[].orig)
EXCL = {d["orig"] for g in filter(None, os.environ.get("EXCLUDE_GOLDEN", "").split(",")) for d in json.load(open(f"{g}/kld.json"))["docs"]}
pools = json.load(open(os.environ.get("POOLS", f"{RXM_WORK}/challenge/pools.json")))["pool"] if not half.lstrip("-").isdigit() else None
if pools is not None: assert len(pools) == len(docs)
half = half if pools is not None else int(half)
tot = collections.Counter(); bycat = collections.defaultdict(list)
for i, d in enumerate(docs):
    if i in EXCL: continue
    if pools is not None:
        if pools[i] != half: continue
    elif half >= 0 and i % 2 != half: continue
    tot[d["category"]] += int(d["tokens"]); bycat[d["category"]].append(i)
allt = sum(tot.values()); rng = random.Random(seed); out = []; got = collections.Counter()
for c in sorted(bycat):
    want = T * tot[c] / allt
    idx = bycat[c][:]; rng.shuffle(idx)
    for i in idx:
        if got[c] >= want: break
        n = min(int(docs[i]["tokens"]), cap); got[c] += n
        out.append(f"{i}:{cap}" if int(docs[i]["tokens"]) > cap else str(i))
print(" ".join(sorted(out, key=lambda s: int(s.split(":")[0]))))
print(f"{len(out)} documents, {sum(got.values())} tokens: " + ", ".join(f"{c} {got[c]}" for c in sorted(got)), file=sys.stderr)
