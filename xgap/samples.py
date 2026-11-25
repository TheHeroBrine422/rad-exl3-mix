#!/usr/bin/env python3
"""samples.py [N=1] -- per xgap source: tags + the first and last characters of N sample documents (for the pilot report)."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
by = collections.defaultdict(list)
for l in open(f"{ROOT}/manifest.jsonl"):
    r = json.loads(l)
    if r["origin"] == "xgap-v1": by[(r["cell"], r["src_id"])].append(r)
for (cell, sid), rows in sorted(by.items()):
    rows.sort(key=lambda r: r["tokens"])
    for r in rows[len(rows) // 2: len(rows) // 2 + N]:
        d = json.loads(open(f"{ROOT}/{r['file']}").readlines()[r["line"]])
        tags = {k: r.get(k) for k in ("kind", "tokens", "length_band", "lang", "licence", "thinking", "has_system", "length", "windowed", "holdout")}
        p = d["prompt"]
        print(f"=== {cell} | {sid} | {r['id']} {json.dumps(tags, ensure_ascii=False)}")
        print(p[:350].replace("\n", "\\n")); print("  [...]"); print(p[-250:].replace("\n", "\\n"))
