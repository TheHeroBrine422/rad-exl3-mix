# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""report181.py -- w181 report tables (read-only on the live state; verdict counts only, RULES-193 amendment 13):
1. confirmation rate per source since my first tag (rate193), 2. my files per allocator / variant / flag (flow181.tally)
with 80 % Wilson intervals, 3. pooled anom / rankreg / xplant over w172's and my period vs the own moves since w172's
first tag."""
import contextlib, io, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow181 as F, rate193 as R  # noqa: E402


def row(name, r):
    n = r[0] - r[4]; k = r[2] + r[3]; lo, hi = R.wilson(k, n)
    return f"| {name} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {100 * k / max(n, 1):.0f} % [{100 * lo:.0f}, {100 * hi:.0f}] |"


H = "| decided | tentative | accept | join | stage 2 pending | confirmed / decided (80 % Wilson) |\n|---|---|---|---|---|---|---|"
print("### Same period (from my first tag)\n"); R.main(None)
print("\n### My files by variant\n\n| allocator / variant / flag " + H)
for k, r in F.tally().items(): print(row(" / ".join(x for x in k if x) , r))
print("\n### Pooled with w172's period (from w172's first tag; file tokens anom / rankreg / xplant of both workers)\n")
b = io.StringIO()
with contextlib.redirect_stdout(b): R.main(1556)
P = {}
for l in b.getvalue().splitlines():
    c = [x.strip() for x in l.split("|")]
    if len(c) < 7 or not c[2].isdigit(): continue
    key = c[1].split(":")[1] if c[1].startswith("w1") else ("own moves" if "own moves" in c[1] else None)
    if key in ("anom", "rankreg", "rankanom", "xplant", "own moves"):
        r = P.setdefault(key, [0] * 5)
        for i in range(5): r[i] += int(c[2 + i])
print("| source " + H)
for k, r in P.items(): print(row(k, r))
