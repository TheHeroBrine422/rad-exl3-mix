#!/usr/bin/env python3
"""auditsum.py AUDIT_DIR NAME... -- #175 (w137) radisa audit report.json (run.sh) summed: finding counts by severity and
rule, split into our kernels (exl3* / qwen35_exl3 objects: names with exl3) and the rest; every non-review finding
listed (kernel, rule, message)."""
import collections, json, sys
d = sys.argv[1]
for n in sys.argv[2:]:
    r = json.load(open(f"{d}/{n}/analysis/report.json"))
    C = collections.Counter(); ex = []
    for k in r["kernels"]:
        name = k.get("name") or k.get("symbol") or "?"
        ours = "exl3" in name.lower()
        for f in k.get("findings", []):
            sev = f.get("severity", "?"); C[(sev, f.get("rule", f.get("code", "?")), "ours" if ours else "stock")] += 1
            if sev not in ("review", "info"):
                ex.append((sev, f.get("rule", f.get("code")), name[:80], (f.get("message") or "")[:160]))
    print(f"## {n}: status {r['status']}, counts {r['finding_counts']}")
    for (sev, rule, who), c in sorted(C.items(), key=lambda x: (x[0][0], -x[1])):
        print(f"  {sev:10s} {rule:28s} {who:6s} {c}")
    for e in ex[:40]:
        print("   ", *e)
