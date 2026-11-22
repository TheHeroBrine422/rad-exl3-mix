#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""table217.py SCORES_DIR GOLDEN EPS.json [--json OUT] -- #217's readout: per size one table, rows = lr x eps (w214's no-eps arm first), columns = KL ratio vs the unhealed
release (ctl-<size>) at 0.25 M and 0.5 M rows (paired per document, 95 % CI, documents better / worse; cmp214's arithmetic) and the 0.25 -> 0.5 M step (0.5 M vs
0.25 M of the same arm). EPS.json: {"2.50": [e_lo, e_mid, e_hi], ...} (RULES217). Arm names: eps arms `<S>E-l<lr>-x<mult>` (S = 25 / 30 / 40, mult 0.3 / 1 / 3), w214's
no-eps arms `<S>A` (lr 1e-3) and `<S>A-lr3e-4`. Score files SCORES_DIR/<arm>-ckpt-2e+05.json.rows (0.25 M) and -5e+05 (0.5 M); a missing read prints `-`."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
import cmp214  # noqa: E402

sd, g, ef = sys.argv[1:4]; out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
eps = json.load(open(ef)); res = {}; md = []
def fmt(d): return f"{d['ratio']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] ({d['docs_better']}/{d['docs_worse']})"
for size, S in (("2.50", "25"), ("3.00", "30"), ("4.00", "40")):
    cf = f"{sd}/ctl-{size}.json.rows"
    if not os.path.exists(cf): continue
    xc, n = cmp214.doc_sums(g, cf)
    md += [f"\n**{size}** (unhealed KL {xc.sum() / n.sum():.6f} / gen row; ratio [95 % CI] (docs better / worse))\n", "| lr | eps | 0.25 M | 0.5 M | 0.25 → 0.5 M step |", "|---|---|---|---|---|"]
    for lr, ls, na in (("1e-3", "1e-3", f"{S}A"), ("3e-4", "3e-4", f"{S}A-lr3e-4")):
        arms = [(na, "none (w214, eager)")] + [(f"{S}E-l{ls}-x{m}", f"{e:.2e} ({m}x)") for m, e in zip(("0.3", "1", "3"), eps[size])]
        for arm, el in arms:
            xs = []
            for ck in ("2e+05", "5e+05"):
                f = f"{sd}/{arm}-ckpt-{ck}.json.rows"
                xs.append(cmp214.doc_sums(g, f)[0] if os.path.exists(f) else None)
            r = {}
            cells = []
            for k, x in zip(("0.25M", "0.5M"), xs):
                if x is None: cells.append("-"); continue
                r[k] = cmp214.paired(xc, x, n); r[k]["kld"] = float(x.sum() / n.sum()); cells.append(fmt(r[k]))
            if xs[0] is not None and xs[1] is not None: r["step"] = cmp214.paired(xs[0], xs[1], n); cells.append(fmt(r["step"]))
            else: cells.append("-")
            res[f"{size}/{arm}"] = r
            md.append(f"| {lr} | {el} | {cells[0]} | {cells[1]} | {cells[2]} |")
print("\n".join(md))
if out: json.dump(res, open(out, "w"), indent=1)
