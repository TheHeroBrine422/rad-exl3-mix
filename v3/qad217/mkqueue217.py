#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkqueue217.py EPS.json OUTDIR -- the 18 arms of RULES217 as four card queues (OUTDIR/q217-card<N>.txt, card217.sh's format; arms dealt round-robin over the
cards in the lead's order: wave 1 = x1 at 2.50 lr 1e-3, 3.00 lr 1e-3, 3.00 lr 3e-4, 4.00 lr 3e-4; then the other x1 arms; then x0.3 and x3; each arm followed by its two served reads, the 0.25 M one first): arm `<S>E-l<lr>-x<mult>`, budgets 2.5e5,5e5, `--lr LR --eps EPS`."""
import json, os, sys
eps = json.load(open(sys.argv[1])); od = sys.argv[2]
S2 = {"2.50": "25", "3.00": "30", "4.00": "40"}; MULT = ("0.3", "1", "3")
def arm(size, lr, m): return (f"{S2[size]}E-l{lr}-x{m}", size, lr, eps[size][MULT.index(m)])
# the lead's order (11-21): wave 1 = the middle eps at 2.50 lr 1e-3, 3.00 lr 1e-3, 3.00 lr 3e-4, 4.00 lr 3e-4 (one a card); then the other middle-eps arms; then 0.3x and 3x
arms = [arm(*x, "1") for x in (("2.50", "1e-3"), ("3.00", "1e-3"), ("3.00", "3e-4"), ("4.00", "3e-4"), ("2.50", "3e-4"), ("4.00", "1e-3"))]
for m in ("0.3", "3"):
    arms += [arm(sz, lr, m) for sz, lr in (("3.00", "1e-3"), ("3.00", "3e-4"), ("2.50", "1e-3"), ("4.00", "3e-4"), ("2.50", "3e-4"), ("4.00", "1e-3"))]
assert len(arms) == 18 and len({a[0] for a in arms}) == 18
q = [[] for _ in range(4)]
for k, (arm, size, lr, e) in enumerate(arms):
    c = q[k % 4]; c.append(f"T {arm} {size} 2.5e5,5e5 --lr {lr} --eps {e:.4e}")
    for ck in ("ckpt-2e+05", "ckpt-5e+05"): c.append(f"R /data/q27/w217 {arm} {size} {ck}")
for n, c in enumerate(q): open(os.path.join(od, f"q217-card{n}.txt"), "w").write("\n".join(c) + "\n")
print(len(arms), "arms;", [sum(l.startswith("T") for l in c) for c in q], "per card")
