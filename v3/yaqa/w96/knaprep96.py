#!/usr/bin/env python3
"""knaprep96.py W94OUT W96OUT -- knap5 re-solves (w94's procedure, KN_HEADGRID on search-gen sims): today's menu and w94's
g-search-gen (GPU headsim) from w94, g96-base (the same plans on score96's CPU sums: control) and g96-more-x (the best
arm), per size: predicted KLD, vs menu, vs g96-base, head pick, plan."""
import ast, json, os, sys
A, B = sys.argv[1:3]
def rd(f):
    r = json.loads(open(f).readline()); h = r.get("head"); h = ast.literal_eval(h) if isinstance(h, str) else h
    if not h:
        pk = r["picks"]; pk = ast.literal_eval(pk) if isinstance(pk, str) else pk; h = {"opt": pk["output.weight"][0]}
    return r["kld"], h
print("| size | arm | predicted KLD | vs menu | vs g96-base | head pick | head bits | plan |")
print("|---|---|---|---|---|---|---|---|")
for s in ("2", "2.5", "3", "4"):
    m = rd(f"{A}/menu-{s}.jsonl")[0]; b = rd(f"{B}/g96-base-{s}.jsonl")[0]
    for name, f in (("menu", f"{A}/menu-{s}.jsonl"), ("w94 g-search-gen", f"{A}/g-search-gen-{s}.jsonl"),
                    ("g96-base", f"{B}/g96-base-{s}.jsonl"), ("g96-more-x", f"{B}/g96-more-x-{s}.jsonl")):
        k, h = rd(f)
        print(f"| {s} | {name} | {k:.6f} | {k / m - 1:+.2%} | {k / b - 1:+.2%} | {h.get('opt')} | {h.get('bits')} | {h.get('plan', '')} |")
