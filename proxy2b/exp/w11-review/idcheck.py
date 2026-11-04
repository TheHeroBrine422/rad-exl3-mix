#!/usr/bin/env python3
"""idcheck.py -- w11 leakage check (CPU, read-only): calib-meta order vs calib.jsonl (the pools' index), each 0.8B
golden's docs: pool, source and tokens against calib-meta[orig]; overlaps between goldens; the windows heal.py,
moments.py and swiglufit use, by pool. Run on the 4070: CUDA_VISIBLE_DEVICES= nice -n 19 python3 -I idcheck.py"""
import json, collections, sys
import numpy as np
META = "/data/challenge/calib-meta"; POOLS = "/workspace/rad-exl3-mix/eval/pools/pools-v3.json"
CORP = "/data/challenge/corpus/calib.jsonl"
m = json.load(open(f"{META}/kld.json")); toks = np.memmap(f"{META}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
pools = json.load(open(POOLS))["pool"]
print("calib-meta docs", len(m["docs"]), "pools", len(pools), "pool counts", collections.Counter(pools))
# calib.jsonl order vs calib-meta order (sources line by line)
try:
    src = []
    with open(CORP) as f:
        for line in f:
            j = json.loads(line); src.append(j.get("source") or j.get("meta", {}).get("source"))
    mism = [i for i, (a, b) in enumerate(zip(src, (d["source"] for d in m["docs"]))) if a != b]
    print(f"calib.jsonl {len(src)} lines; source mismatches vs calib-meta: {len(mism)} {mism[:5]}; keys {list(j.keys())}")
except FileNotFoundError:
    print("no", CORP)
G = {}
for g in sys.argv[1:] or ["/data/g08-search1", "/data/g08-fit1"]:
    gm = json.load(open(f"{g}/kld.json")); gt = np.memmap(f"{g}/tokens.i32", np.int32, "r")
    goff = np.concatenate([[0], np.cumsum([d["tokens"] for d in gm["docs"]])])
    bad_src = bad_tok = 0
    for k, d in enumerate(gm["docs"]):
        i = d["orig"]; c = m["docs"][i]
        bad_src += d["source"] != c["source"]
        bad_tok += not np.array_equal(gt[goff[k]:goff[k + 1]], toks[off[i]:off[i] + d["tokens"]])
    G[g] = {d["orig"] for d in gm["docs"]}
    print(f"{g}: {len(gm['docs'])} docs, pools {collections.Counter(pools[d['orig']] for d in gm['docs'])}, "
          f"source mismatch {bad_src}, token mismatch {bad_tok}, max tokens {max(d['tokens'] for d in gm['docs'])}, "
          f"score_from>0 {sum(d['score_from'] > 0 for d in gm['docs'])}")
ks = list(G)
for a in range(len(ks)):
    for b in range(a + 1, len(ks)): print(f"overlap {ks[a]} & {ks[b]}: {len(G[ks[a]] & G[ks[b]])}")
# moments.py windows (its code, --pools qad --tokens 1e6 --win 2048 --seed 0)
want = {"qad"}; wins = []
for i, d in enumerate(m["docs"]):
    if pools[i] not in want: continue
    n = d["tokens"]
    if n < 2048:
        if n >= 64: wins.append((int(off[i]), n, i))
    else: wins += [(int(off[i] + s), 2048, i) for s in range(0, n - 2048 + 1, 2048)]
rng = np.random.default_rng(0); order = rng.permutation(len(wins))  # shuffle as moments.py (list shuffle == permutation? check below)
w2 = [(a, b) for a, b, _ in wins]; np.random.default_rng(0).shuffle(w2)
cum = np.cumsum([n for _, n in w2]); w2 = w2[:int(np.searchsorted(cum, 1_000_000)) + 1]
docof = {}
for a, b, i in wins: docof[a] = i
md = {docof[a] for a, _ in w2}
print(f"moments windows: {len(w2)}, docs {len(md)}, pools {collections.Counter(pools[i] for i in md)}, "
      f"in goldens {[len(md & G[g]) for g in ks]}")
