#!/usr/bin/env python3
"""vocabtier.py CALIB.jsonl TOKENIZER_DIR OUT.json -- groundwork for the vocabulary-tiered head/embedding (O1 in
research/quant-sota.md). Over the calib corpus, the pools in VT_POOLS (default fit,search,qad: verify excluded; pool
file POOLS=eval/pools/pools.json):
  - input-token frequency per vocab id (what the embedding sees)
  - next-token frequency at scored positions (a proxy for E[p_v], the head's output Fisher diagonal)
and the coverage curve: what share of the mass the top N ids hold, for N in a ladder, plus the bytes a 2/3-tier
head would cost at given K splits (EXL3 head = 248320 x 5120; K bits per weight + ~0.3% scale overhead).
Tokenises with the HF tokenizer.json (tokenizers library) of TOKENIZER_DIR."""
import json, os, sys, collections
import numpy as np
from tokenizers import Tokenizer
CORP, TOK, OUT = sys.argv[1:4]
pools = json.load(open(os.environ.get("POOLS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../eval/pools/pools.json"))))["pool"]
USE = set(os.environ.get("VT_POOLS", "fit,search,qad").split(","))
tk = Tokenizer.from_file(f"{TOK}/tokenizer.json")
V = 248320
cnt = np.zeros(V, np.int64)
docs = [json.loads(l) for l in open(CORP)]
texts = []
for i, d in enumerate(docs):
    if pools[i] not in USE: continue
    t = d.get("prompt") or d.get("text") or ""
    if isinstance(d.get("messages"), list): t = "\n".join(str(m.get("content", "")) for m in d["messages"])
    t = t + (d.get("completion") or d.get("response") or "")
    texts.append(t)
for b in range(0, len(texts), 256):
    for e in tk.encode_batch(texts[b:b + 256], add_special_tokens=False):
        ids = np.asarray(e.ids); ids = ids[ids < V]; np.add.at(cnt, ids, 1)
tot = cnt.sum(); order = np.argsort(-cnt); cum = np.cumsum(cnt[order]) / tot
ladder = [1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, V]
cov = {n: float(cum[n - 1]) for n in ladder}
nz = int((cnt > 0).sum())
ROWB = 5120 / 8 * 1.003                       # bytes per row per bit
PARAMS = 26895998464
def head_bpw(split):                          # [(rows, K)] -> real-bpw cost of the head
    return sum(r * k * ROWB for r, k in split) * 8 / PARAMS
plans = {}
for core in (5000, 10000, 20000, 30000, 50000):
    for kc, kt in ((6, 3), (6, 2), (5, 3), (5, 2), (8, 3), (8, 4)):
        plans[f"top{core}@K{kc}+rest@K{kt}"] = {"bpw": head_bpw([(core, kc), (V - core, kt)]),
                                                "tail_mass": float(1 - cum[core - 1])}
for k in (3, 4, 5, 6, 8): plans[f"uniform K{k}"] = {"bpw": head_bpw([(V, k)]), "tail_mass": 0.0}
json.dump({"tokens": int(tot), "docs": len(texts), "ids_seen": nz, "coverage": cov, "plans": plans,
           "top_ids": order[:200].tolist()}, open(OUT, "w"), indent=1)
np.save(OUT.replace(".json", "-counts.npy"), cnt)
print(f"{tot} tokens over {len(texts)} docs; {nz} of {V} ids ever seen ({nz / V:.1%})")
for n in ladder: print(f"  top {n:6d} ids: {cov[n]:.4%} of tokens")
for k, v in plans.items(): print(f"  {k:24} head {v['bpw']:.4f} bpw, tail mass {v['tail_mass']:.4%}")
