#!/usr/bin/env python3
"""ledger_add.py ROLES.json POOLS.json LEDGER.jsonl [--date YYYY-MM-DD] [--all USE DATE] [GOLDEN_DIR_OR_KLD_JSON ...] -- append uses to the
document-use ledger without a full rebuild (mkledger.py needs every golden on one machine).

Goldens: one line per document from kld.json (docs[].orig, score_from, tokens), use = the golden's name (its directory
name, or the file name without `-kld.json` / `.json` for a copied kld.json), role from ROLES (exact name or a "prefix*"
entry, as mkledger.py). Only goldens cut from the calib corpus (corpus_fnv1a64 must match CALIB_FNV, default the
calib corpus's) are accepted. `--all USE DATE`: one line with "doc": "all" (a collection over every calib token).
A (doc, use) already in the ledger is skipped, so re-running is safe. Prints what it added per pool.
`--date`: the date of the use for golden rows (default: the kld.json's "created", which for a slice cut from the
published calib golden is that golden's 2026-10-25, not the date of the use; w141 audit 11-13)."""
import collections, fnmatch, json, os, sys

CALIB_FNV = os.environ.get("CALIB_FNV", "f3fe4bf9488a5616")
ROLES, POOLS, LED = sys.argv[1:4]; rest = sys.argv[4:]
roles = json.load(open(ROLES)); pool = json.load(open(POOLS))["pool"]
have = set()
for l in open(LED):
    r = json.loads(l); have.add((r["doc"], r["use"]))


def role(name):
    if name in roles: return roles[name]
    for k, v in roles.items():
        if "*" in k and fnmatch.fnmatch(name, k): return v
    raise SystemExit(f"no role for {name}: add it to {ROLES}")


out, alls, files = [], [], []
use_date = None
i = 0
while i < len(rest):
    if rest[i] == "--all": alls.append((rest[i + 1], rest[i + 2])); i += 3
    elif rest[i] == "--date": use_date = rest[i + 1]; i += 2
    else: files.append(rest[i]); i += 1


def rec(doc, use, r, a, b, date):
    if (doc, use) in have: return False
    out.append({"doc": doc, "use": use, "role": r["role"], "line": r["line"], "decisions": r["decisions"],
                "from": a, "to": b, "date": date, "note": r["note"]}); have.add((doc, use)); return True


for use, date in alls:
    rec("all", use, role(use), None, None, date)
for f in files:
    p = f if f.endswith(".json") else os.path.join(f, "kld.json")
    name = os.path.basename(f.rstrip("/")) if not f.endswith(".json") else os.path.basename(f).removesuffix(".json").removesuffix("-kld")
    m = json.load(open(p))
    if m.get("corpus_fnv1a64") != CALIB_FNV: raise SystemExit(f"{name}: corpus {m.get('corpus')} {m.get('corpus_fnv1a64')} is not calib")
    r = role(name); date = use_date or m.get("created", "")[:10]; c = collections.Counter()
    for d in m["docs"]:
        if rec(d["orig"], name, r, d["score_from"], d["tokens"] - 1, date): c[pool[d["orig"]]] += 1
    print(f"{name}: +{sum(c.values())} docs {dict(c)} role {r['role']} line {r['line']}")
with open(LED, "a") as fh:
    for o in out: fh.write(json.dumps(o) + "\n")
print(f"appended {len(out)} lines")
