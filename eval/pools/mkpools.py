#!/usr/bin/env python3
"""mkpools.py CALIB.jsonl GOLDEN_DIR... -> pools.json: the fixed assignment of every calib document to one pool
(fit / search / qad / verify), see docs/ (data discipline). Run once; the output is the single source of truth.

- Documents already used in the calib-A slice stay in fit, calib-B / calib-B-big ones stay in verify (golden kld.json
  docs[].orig); the remaining documents are drawn at random (seed 0) per category so that each pool ends up with its
  token share of every category.
- The 8 long agentic traces (~250k tokens) are assigned by hand: 2 verify, 2 search, 4 qad. Four of them
  (click, requests -> qad; ripgrep, cobra -> search) had their first 8k tokens in the legacy golden-calB/-calBbig
  slices: those slices stay valid for comparisons made before any search/QAD, but verification after search/QAD
  must use slices built from the verify pool only.
"""
import json, random, sys, collections

SHARE = {"fit": 0.12, "search": 0.12, "qad": 0.58, "verify": 0.18}
TRACES = {"agent-trace/rich": "verify", "agent-trace/express": "verify",
          "agent-trace/ripgrep": "search", "agent-trace/cobra": "search",
          "agent-trace/attrs": "qad", "agent-trace/lua": "qad",
          "agent-trace/click": "qad", "agent-trace/requests": "qad"}

corp = sys.argv[1]
docs = [json.loads(l) for l in open(corp)]
pool = {}
for i, d in enumerate(docs):
    if d["source"] in TRACES: pool[i] = TRACES[d["source"]]
assert len(pool) == 8, pool
for g in sys.argv[2:]:
    p = "fit" if g.rstrip("/").endswith("calA") else "verify"
    for d in json.load(open(f"{g}/kld.json"))["docs"]:
        pool.setdefault(d["orig"], p)

rng = random.Random(0)
bycat = collections.defaultdict(list)
for i, d in enumerate(docs): bycat[d["category"]].append(i)
for c in sorted(bycat):
    tot = sum(int(docs[i]["tokens"]) for i in bycat[c])
    got = collections.Counter()
    for i in bycat[c]:
        if i in pool: got[pool[i]] += int(docs[i]["tokens"])
    free = [i for i in bycat[c] if i not in pool]; rng.shuffle(free)
    for i in free:   # give each document to the pool furthest below its target share
        p = min(SHARE, key=lambda q: got[q] / tot - SHARE[q])
        pool[i] = p; got[p] += int(docs[i]["tokens"])

out = {"corpus": corp, "n_docs": len(docs), "shares": SHARE, "traces": TRACES,
       "pool": [pool[i] for i in range(len(docs))]}
json.dump(out, sys.stdout)
n = collections.Counter(out["pool"]); t = collections.Counter()
for i, p in enumerate(out["pool"]): t[p] += int(docs[i]["tokens"])
print("\n" + "  ".join(f"{p}: {n[p]} docs {t[p]/1e6:.2f}M tok" for p in SHARE), file=sys.stderr)
for c in sorted(bycat):
    ct = collections.Counter(); tt = sum(int(docs[i]["tokens"]) for i in bycat[c])
    for i in bycat[c]: ct[out["pool"][i]] += int(docs[i]["tokens"])
    print(f"  {c:13} " + " ".join(f"{p} {ct[p]/tt:4.0%}" for p in SHARE), file=sys.stderr)
