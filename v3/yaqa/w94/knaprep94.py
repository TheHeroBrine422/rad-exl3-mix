#!/usr/bin/env python3
"""knaprep94.py OUTDIR -- w88's knap5 head grid re-solved on w94's headsim rows (accurate f16, lambda 0): per size the
head pick and the predicted KLD of menu (today's menu), old37 (w88's 11-01 curves, donors K3-K7 only) and g-<rowset>
(the corrected-rowerr plans of today's K3-K7 donors priced on search-all / search-gen / qad-gen rows)."""
import ast, json, os, sys

O = sys.argv[1]; rows = []
for s in ("2", "2.5", "3", "4"):
    m = None
    for a in ("menu", "old37", "g-search-all", "g-search-gen", "g-qad-gen"):
        f = f"{O}/{a}-{s}.jsonl"
        if not os.path.exists(f): continue
        r = json.loads(open(f).readline()); h = r.get("head")
        h = ast.literal_eval(h) if isinstance(h, str) else h
        if not h:
            pk = r["picks"]; pk = ast.literal_eval(pk) if isinstance(pk, str) else pk
            h = {"opt": pk["output.weight"][0]}
        if a == "menu": m = r["kld"]
        hk = h.get("head_kl") if isinstance(h, dict) else None
        rows.append((s, a, r["kld"], (r["kld"] / m - 1) if m else 0.0, h.get("opt") if isinstance(h, dict) else h,
                     h.get("bits") if isinstance(h, dict) else None, hk, h.get("plan", "") if isinstance(h, dict) else ""))
print("| size | arm | predicted KLD | vs menu | head pick | head bits | head-only KL | plan |")
print("|---|---|---|---|---|---|---|---|")
for s, a, k, d, o, b, hk, p in rows:
    print(f"| {s} | {a} | {k:.6f} | {d:+.2%} | {o} | {b} | {hk if hk is None else f'{hk:.2e}'} | {p} |")
