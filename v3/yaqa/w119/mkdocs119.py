#!/usr/bin/env python3
"""mkdocs119.py KLD.json GEN.u8 POOLS.json LEDGER.jsonl OUT.json -- documents and row windows of w119's head-input dumps
(E2 step 0, RULES.md), fixed before any data. Same window rule as w94's mkdocs94.py (one window of consecutive golden
rows inside the first CAP tokens, seeded start), WIN 256 rows.

  search (fresh, decision set of E2 / E3 (c)): search-pool documents in no golden-search27* slice and not in w94's dump
          (214 on 11-12); seeded shuffle (seed 119), the first 150. The other 64 are kept for E3 (d)'s slice ("search-rest").
  fit (pricing / plan choice): 200 fit-pool documents, seeded shuffle (seed 119).
  verify (final check only): 100 clean verify documents (in no golden: only '@' collections), < 32,768 tokens, seed 119.
Documents whose window would hold no gen row are skipped (the next one in the shuffle is taken)."""
import json, sys, numpy as np

KLD, GEN, POOLS, LED, OUT = sys.argv[1:6]
WIN, CAP, SEED, LONG = 256, 8192, 119, 32768
NEED = {"search": 150, "fit": 200, "verify": 100}
m = json.load(open(KLD)); gen = np.fromfile(GEN, np.uint8); pool = json.load(open(POOLS))["pool"]
assert len(pool) == len(m["docs"]) and len(gen) == m["rows"]
uses = {}
for r in map(json.loads, open(LED)): uses.setdefault(r["doc"], set()).add(r["use"])


def window(d):
    D = m["docs"][d]; n = min(D["tokens"] - 1, CAP) - D["score_from"]
    if n <= 0: return None
    w = min(WIN, n); a = int(np.random.default_rng(SEED + d).integers(0, n - w + 1))
    g = int(gen[D["row0"] + a:D["row0"] + a + w].sum())
    return {"doc": d, "pool": pool[d], "source": D["source"], "row0": D["row0"] + a, "k0": a, "rows": w, "gen": g,
            "pos0": D["score_from"] + a, "fwd": D["score_from"] + a + w}


def pick(cands, n):
    out = []
    for d in np.random.default_rng(SEED).permutation(np.array(cands)):
        x = window(int(d))
        if x and x["gen"] > 0: out.append(x)
        if len(out) == n: break
    return out


fresh = [d for d in range(len(pool)) if pool[d] == "search"
         and not any(u.startswith("golden-search27") or u.startswith("w94") for u in uses.get(d, ()))]
fit = [d for d in range(len(pool)) if pool[d] == "fit"]
clean = [d for d in range(len(pool)) if pool[d] == "verify" and all(u.startswith("@") for u in uses.get(d, ()))
         and m["docs"][d]["tokens"] < LONG]
out = {"win": WIN, "cap": CAP, "seed": SEED, "sets": {}, "pools": {"fresh_search": len(fresh), "fit": len(fit), "clean_verify": len(clean)}}
out["sets"]["search"] = pick(fresh, NEED["search"])
used = {x["doc"] for x in out["sets"]["search"]}
out["search_rest"] = sorted(int(d) for d in fresh if d not in used)
out["sets"]["fit"] = pick(fit, NEED["fit"])
out["sets"]["verify"] = pick(clean, NEED["verify"])
for k, v in out["sets"].items():
    print(k, len(v), "docs", sum(x["rows"] for x in v), "rows", sum(x["gen"] for x in v), "gen rows",
          sum(x["fwd"] for x in v), "tokens forwarded")
print("search_rest", len(out["search_rest"]), "pools", out["pools"])
json.dump(out, open(OUT, "w"), indent=0)
