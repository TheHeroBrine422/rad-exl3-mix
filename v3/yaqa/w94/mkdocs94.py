#!/usr/bin/env python3
"""mkdocs94.py KLD.json GEN.u8 POOLS.json LEDGER.jsonl OUT.json -- the documents and row windows of w94's head-input
dumps (#124), fixed before any data.

  search (held out): every calib document of the retired golden-search27 (all 103 are search-pool docs; reusing them
          spends no fresh search documents)
  qad (fit / pricing): qad-pool documents in a seeded shuffle (seed 94) until the windows hold >= NEED gen rows
Per document one window of up to WIN consecutive golden rows inside the first CAP tokens (row k = position
score_from + k, predicting token score_from + k + 1), start drawn uniformly (seed 94 + doc). Every row of the window is
dumped with its gen flag, so the same documents give a gen-only set and a consecutive 'prefill-like' set."""
import json, sys, numpy as np

KLD, GEN, POOLS, LED, OUT = sys.argv[1:6]
WIN, CAP, NEED, SEED = 512, 8192, 32768, 94
m = json.load(open(KLD)); gen = np.fromfile(GEN, np.uint8); pool = json.load(open(POOLS))["pool"]
assert m["corpus_fnv1a64"] == "f3fe4bf9488a5616" and len(pool) == len(m["docs"]) and len(gen) == m["rows"]
s27 = sorted({r["doc"] for r in map(json.loads, open(LED)) if r["use"] == "golden-search27"})
assert all(pool[d] == "search" for d in s27)


def window(d):
    D = m["docs"][d]; n = min(D["tokens"] - 1, CAP) - D["score_from"]      # rows whose position is < CAP
    if n <= 0: return None
    w = min(WIN, n); a = int(np.random.default_rng(SEED + d).integers(0, n - w + 1))
    g = int(gen[D["row0"] + a:D["row0"] + a + w].sum())
    return {"doc": d, "pool": pool[d], "source": D["source"], "row0": D["row0"] + a, "k0": a, "rows": w, "gen": g,
            "pos0": D["score_from"] + a, "fwd": D["score_from"] + a + w}          # forward tokens [0, fwd)


out = {"win": WIN, "cap": CAP, "need": NEED, "seed": SEED, "sets": {}}
out["sets"]["search"] = [x for x in map(window, s27) if x]
qad = [d for d in range(len(pool)) if pool[d] == "qad"]
order = np.random.default_rng(SEED).permutation(qad); sel, g = [], 0
for d in order:
    x = window(int(d))
    if x and x["gen"] > 0: sel.append(x); g += x["gen"]
    if g >= NEED: break
out["sets"]["qad"] = sel
for k, v in out["sets"].items():
    print(k, len(v), "docs", sum(x["rows"] for x in v), "rows", sum(x["gen"] for x in v), "gen rows",
          sum(x["fwd"] for x in v), "tokens forwarded")
json.dump(out, open(OUT, "w"), indent=0)
