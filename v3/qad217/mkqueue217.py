#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkqueue217.py EPS.json OUTDIR -- the 18 arms of RULES217 as four card queues (OUTDIR/q217-card<N>.txt, card217.sh's format; arms dealt round-robin over the
cards in the order size x lr x eps multiplier, each arm followed by its two served reads): arm `<S>E-l<lr>-x<mult>`, budgets 2.5e5,5e5, `--lr LR --eps EPS`."""
import json, os, sys
eps = json.load(open(sys.argv[1])); od = sys.argv[2]
arms = []
for size, S in (("4.00", "40"), ("3.00", "30"), ("2.50", "25")):
    for lr in ("1e-3", "3e-4"):
        for m, e in zip(("0.3", "1", "3"), eps[size]): arms.append((f"{S}E-l{lr}-x{m}", size, lr, e))
q = [[] for _ in range(4)]
for k, (arm, size, lr, e) in enumerate(arms):
    c = q[k % 4]; c.append(f"T {arm} {size} 2.5e5,5e5 --lr {lr} --eps {e:.4e}")
    for ck in ("ckpt-2e+05", "ckpt-5e+05"): c.append(f"R /data/q27/w217 {arm} {size} {ck}")
for n, c in enumerate(q): open(os.path.join(od, f"q217-card{n}.txt"), "w").write("\n".join(c) + "\n")
print(len(arms), "arms;", [sum(l.startswith("T") for l in c) for c in q], "per card")
