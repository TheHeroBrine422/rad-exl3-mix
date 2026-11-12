#!/usr/bin/env python3
"""ledger119.py DOCS119.json OUTDIR -- ledger rows (eval/pools/ledger.jsonl format, w94's from / to = the dumped window
pos0..fwd) and roles of w119's three head dumps, for the lead to merge (ledger files are shared)."""
import json, sys

D, OUT = sys.argv[1:3]
N = {"search": ("select", 2, "#155 / #156 (w119, 2026-11-12): bf16 head inputs (final-norm output), one <= 256-row window per document inside the first 8,192 tokens (gen rows scored), 150 fresh search-pool docs (no golden-search27*, not in w94); E2 order / sigma_out and E3 layout decisions (v3/yaqa/w119/RULES.md)"),
     "fit": ("fit", 2, "#156 / #157 (w119, 2026-11-12): bf16 head inputs, one <= 256-row window per document, 200 fit-pool docs; E3 plan choice and E9 head pricing (fit-gen, calib-weighted)"),
     "verify": ("check", 1, "#156 (w119, 2026-11-12): bf16 head inputs, one <= 256-row window per document, 100 clean verify-pool docs (no golden); E3's single final check, read once")}
sets = json.load(open(D))["sets"]; roles, out = {}, []
for s, (role, dec, note) in N.items():
    use = f"w119-headx-{s}"; roles[use] = {"role": role, "line": "v3", "decisions": dec, "note": note}
    for w in sets[s]:
        out.append({"doc": w["doc"], "use": use, "role": role, "line": "v3", "decisions": dec, "from": w["pos0"], "to": w["fwd"],
                    "date": "2026-11-12", "note": note})
open(f"{OUT}/ledger-w119.jsonl", "w").write("".join(json.dumps(o) + "\n" for o in out))
json.dump(roles, open(f"{OUT}/ledger-roles-w119.json", "w"), indent=1)
print(len(out), {k: sum(o["use"] == k for o in out) for k in roles})
