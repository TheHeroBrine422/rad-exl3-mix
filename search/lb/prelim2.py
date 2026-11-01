#!/usr/bin/env python3
"""prelim2.py RESULTS.jsonl TAG... -- a prelim KLD table (format of the removed docs/prelim-exl3-v2-kld.md, in git history) for held-out ver1 results.

TAGs: w-... mixes (one per row). Columns: real bpw, ver1 KLD, top-1, the nearest IQR-V3 entries below / above in
size (leaderboard), and 'vs leaderboard frontier': our ver1 KLD mapped to the leaderboard via the same EXL3-V1
containers on ver1 (w-ref-*; ratio leaderboard / ver1, linear in bpw) over the Pareto front of all other entries
(log-linear between points) at our size. Also 'vs unsloth': the same over the UD frontier (the category score)."""
import json, math, os, re, sys
import numpy as np
LBF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results-2026-10-30.json")
lb = json.load(open(LBF)); P = lb["params"]
ent = [e for e in lb["entries"] if e.get("scores", {}).get("kld_mean") is not None]
res = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if "kld" in r and r.get("bytes"): res[r["tag"]] = r
bpw = lambda r: r["bytes"] * 8 / P
by = {e["name"]: e for e in ent}


def lbname(c):
    m = re.fullmatch(r"(sc|plain)(\d\.\d\d)-e\d", c)
    return None if not m else f"EXL3-{'SC-' if m.group(1) == 'sc' else ''}{m.group(2)}bpw"


cal = sorted((bpw(r), by[lbname(t[6:])]["scores"]["kld_mean"] / r["kld"]) for t, r in res.items()
             if t.startswith("w-ref-") and lbname(t[6:]) in by)


def front(pts):
    out = []
    for b, k in sorted(pts):
        if not out or k < out[-1][1]: out.append((b, k))
    return out


def loginterp(pts, b):
    i = 0 if b <= pts[0][0] else len(pts) - 2 if b >= pts[-1][0] else max(j for j in range(len(pts) - 1) if pts[j][0] <= b)
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return math.exp(math.log(y0) + (b - x0) / (x1 - x0) * (math.log(y1) - math.log(y0)))


others = front([(e["bytes"] * 8 / P, e["scores"]["kld_mean"]) for e in ent if e["author"] != "Hero422" and e["group"] != "official"])
ud = front([(e["bytes"] * 8 / P, e["scores"]["kld_mean"]) for e in ent if e["group"] == "ud"])
iqr = sorted((e["bytes"] * 8 / P, e["scores"]["kld_mean"], e["name"]) for e in ent if e["group"] == "iqr3")
print("| real bpw | KLD | top-1 | nearest IQR-V3 below | nearest IQR-V3 above | vs leaderboard frontier | vs unsloth |")
print("|---|---|---|---|---|---|---|")
for t in sys.argv[2:]:
    if t not in res: print(f"| {t}: not measured yet |"); continue
    r = res[t]; b = bpw(r); est = r["kld"] * float(np.interp(b, [c[0] for c in cal], [c[1] for c in cal]))
    lo = max((x for x in iqr if x[0] <= b), default=None); hi = min((x for x in iqr if x[0] > b), default=None)
    f = lambda x: f"{x[2]} ({x[0]:.2f}, {x[1]:.5f})" if x else "—"
    print(f"| {b:.2f} | {r['kld']:.5f} | {r['top1']:.1%} | {f(lo)} | {f(hi)} | {est / loginterp(others, b):.2f}× | {est / loginterp(ud, b):.2f}× |")
