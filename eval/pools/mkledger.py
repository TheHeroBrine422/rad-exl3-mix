#!/usr/bin/env python3
"""mkledger.py ROLES.json POOLS.json CALIB_GOLDEN OUT.jsonl GOLDEN_DIR... -- the document-use ledger (JSONL).

One line per (calib document, use):
  {"doc": i, "use": "golden-calA", "role": ..., "line": ..., "decisions": ..., "from": p0, "to": p1, "date": ..., "note": ...}
from/to: the scored position range [from, to) of the document in that use (null = whole document / not position-bound).
Uses: every calib golden (docs[].orig, score_from, tokens; ROLES key = golden name, or a "prefix*" entry), "@yaqa"
(the qad windows the YAQA collections read), "@headsens" (qad-pool head Fisher); "@hessians-calib" is one line with
"doc": "all". Public goldens (no orig into calib) are skipped. Append future uses by re-running (goldens) or by
hand-appending lines for non-golden uses (e.g. QAD training: role train, one line per document trained on);
ledger_add.py appends goldens and whole-calib collections without a full rebuild.
Query example (eligible verify docs for comparing two v3 arms): docs with no line whose role is train / fit-hessian /
fit-diag / select / monitor for a line touching v3, ordered by summed decisions."""
import datetime, fnmatch, json, os, sys
import numpy as np
ROLES, POOLS, CG, OUT = sys.argv[1:5]; GOLD = sys.argv[5:]
roles = json.load(open(ROLES)); pool0 = json.load(open(POOLS))["pool"]
def role(name):
    if name in roles: return roles[name]
    for k, v in roles.items():
        if "*" in k and fnmatch.fnmatch(name, k): return v
    raise SystemExit(f"no role for {name}: add it to {ROLES}")
out = []
def rec(doc, use, r, a=None, b=None, date=None):
    out.append({"doc": doc, "use": use, "role": r["role"], "line": r["line"], "decisions": r["decisions"],
                "from": a, "to": b, "date": date, "note": r["note"]})
for g in GOLD:
    m = json.load(open(f"{g}/kld.json")); name = os.path.basename(g.rstrip("/"))
    if not m["docs"] or "orig" not in m["docs"][0] or "calib" not in m.get("corpus", "calib"): continue
    r = role(name); date = datetime.datetime.fromtimestamp(os.path.getmtime(f"{g}/kld.json"), datetime.timezone.utc).strftime("%Y-%m-%d")
    for d in m["docs"]: rec(d["orig"], name, r, d["score_from"], d["tokens"] - 1, date)
# YAQA: first 200 windows of the seed-0 shuffle of 2048-token qad windows (collect.py with pools.json)
meta = json.load(open(f"{CG}/kld.json")); docs = meta["docs"]
off = np.cumsum([0] + [d["tokens"] for d in docs]); wins = []
for i, d in enumerate(docs):
    if pool0[i] != "qad": continue
    for a in range(0, d["tokens"] - 2048 + 1, 2048): wins.append((int(off[i] + a), i, a))
ref = list(wins); np.random.default_rng(0).shuffle(ref)      # exactly collect.py's rng.shuffle(wins)
for _, i, a in sorted(ref[:200], key=lambda w: (w[1], w[2])): rec(i, "@yaqa", roles["@yaqa"], a, a + 2048, "2026-10-31")
for i, p in enumerate(pool0):
    if p == "qad": rec(i, "@headsens", roles["@headsens"], None, None, "2026-10-31")
rec("all", "@hessians-calib", roles["@hessians-calib"])
with open(OUT, "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
n = len({r["doc"] for r in out if r["doc"] != "all"})
print(f"{len(out)} lines, {n} distinct docs; by use: " +
      ", ".join(f"{u} {sum(1 for r in out if r['use'] == u)}" for u in dict.fromkeys(r["use"] for r in out)))
