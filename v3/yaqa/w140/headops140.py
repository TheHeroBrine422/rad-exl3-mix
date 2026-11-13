#!/usr/bin/env python3
"""headops140.py RES TAG... -- #162-02 (w140): from prof140.sh's serve-prof-TAG.log (radiance --profile-ops table, parsed
as profsum176.py does), the head's op rows (output.weight / logits / lm_head, not blk.*) and all rows: calls, total ms,
ms per head call. Prints one JSON line per TAG."""
import json, re, sys


def table(path):
    lines = open(path, errors="replace").read().splitlines()
    st = [i for i, l in enumerate(lines) if l.startswith("per-op timing")]
    rows = {}
    for l in (lines[st[-1]:] if st else []):
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\w+)\s+(\S+)\s+([\d.]+|-)\s+(.*)", l)
        if m:
            c, t = rows.get(m.group(7).strip(), (0, 0.0)); rows[m.group(7).strip()] = (c + int(m.group(1)), t + float(m.group(2)))
    return rows


def ishead(op):
    return bool(re.search(r"(?<![\w.])output\.weight", op)) or "logits" in op.lower() or "lm_head" in op.lower()


res = sys.argv[1]
for tag in sys.argv[2:]:
    rows = table(f"{res}/serve-prof-{tag}.log")
    head = {k: v for k, v in rows.items() if ishead(k)}
    hc = max((c for c, _ in head.values()), default=0); ht = sum(t for _, t in head.values())
    print(json.dumps({"tag": tag, "head_ops": {k: [c, round(t, 3)] for k, (c, t) in head.items()}, "head_calls": hc, "head_ms": round(ht, 3),
                      "head_ms_per_call": round(ht / hc, 4) if hc else None, "all_ms": round(sum(t for _, t in rows.values()), 1), "n_ops": len(rows)}))
