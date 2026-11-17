# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""p175filter.py CANDS.json OUT.json [--report OUT.md] -- the phase-2 candidate set (RULES-picks am. 11-16 / 11-17).
On the phase-1 corpus (s195c2-c5, c:4151 left out), size-adjusted with the frozen measured b (p2lib.measured_b), each
candidate vs the size's best (paired per-document bootstrap 2,000):
  round sizes 4.5 / 6.0 / 6.5 / 7.0 (am. 11-17 1:50 AM): keep every candidate within 1.5 % of the best or whose 95 % CI vs
    the best includes 0; at least 3, at most 8, closest to the best first;
  other sizes (am. 11-16 5:15 PM §2): the top 5.
Candidates without a full phase-1 read are listed and left out (re-run once scored). OUT = CANDS's format, only kept ones,
each with "p1": {"vs_best_pct", "lo", "hi", "rank"} and the frozen b in OUT["_b"]."""
import json, sys

import p2lib as L

cp, outp = sys.argv[1:3]; rep = sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else None
C = json.load(open(cp)); R = L.Reads(1); keep = L.keep_mask(1)
B = L.measured_b(R, C, keep)
ROUND = {4.5, 6.0, 6.5, 7.0}
out, lines, missing = {"_b": B}, [], []
for s in sorted(C, key=float):
    S = float(s); b = B.get(S, 0.0)
    v = [c for c in C[s] if R.full(c["tag"])]
    missing += [c["tag"] for c in C[s] if not R.full(c["tag"])]
    if not v: continue
    score = {c["tag"]: float(__import__("math").log(R.vec(c["tag"], keep).sum())) + b * (R.bpw[c["tag"]] - S) for c in v}
    v.sort(key=lambda c: score[c["tag"]]); best = v[0]["tag"]; rb = R.vec(best, keep)
    rows = []
    for k, c in enumerate(v):
        t = c["tag"]
        if t == best: e, lo, hi = 0.0, 0.0, 0.0
        else: e, lo, hi = L.boot(R.vec(t, keep), rb, b * (R.bpw[t] - R.bpw[best]))
        rows.append((c, e, lo, hi, k))
    if S in ROUND:
        sel = [r for r in rows if r[1] <= 0.015 or r[2] <= 0.0]
        if len(sel) < 3: sel = rows[:3]
        sel = sel[:8]
    else: sel = rows[:5]
    out[s] = []
    for c, e, lo, hi, k in sel:
        c = dict(c); c["p1"] = {"vs_best_pct": L.pct(e), "lo": L.pct(lo), "hi": L.pct(hi), "rank": k + 1}; out[s].append(c)
    lines.append(f"| {S:g} | {len(v)} | {len(sel)} | b {b:.2f} | " + ", ".join(f"{c['tag'].split('-', 2)[-1]} {L.pct(e):+.2f}" for c, e, lo, hi, k in sel) + " |")
json.dump(out, open(outp, "w"))
txt = "| size | scored | kept | b | kept (vs best, phase-1 corpus, c:4151 out) |\n|---|---|---|---|---|\n" + "\n".join(lines)
txt += f"\n\nnot fully scored (left out): {len(missing)}" + (": " + ", ".join(missing) if missing else "")
print(txt)
if rep: open(rep, "w").write(txt + "\n")
