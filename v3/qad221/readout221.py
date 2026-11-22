#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""readout221.py [W221_DIR] [--json OUT] -- w221's table (box2 /data/q27/w221): per run x {0.25 M, 0.5 M} on j217: ratio vs the unhealed release (paired per document, 95 % CI, docs better / worse),
paired vs the run's control (25C for the 2.50 runs, C30 for L32-30), ratio vs unhealed and vs control without w217's blow-up documents (calib ids 1893, 3820), documents whose KL / row
rises > 10x (vs unhealed or vs the control; min 8 gen rows), w217's blow-up rule (>= 0.3 and >= 5x unhealed), s / step (median of the main steps' wall). Markdown to stdout."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
import cmp214

G = "/data/q27/w217/slices/golden-j217"; S17 = "/data/q27/w217/scores"; BLOW = {1893, 3820}
RUNS = [  # label, size, run dir root, scores root, control label
    ("25C", "2.50", "/data/q27/w220", "/data/q27/w220/scores", None),
    ("L32", "2.50", None, None, "25C"), ("L48", "2.50", None, None, "25C"), ("T128", "2.50", None, None, "25C"), ("CB", "2.50", None, None, "25C"),
    ("C30", "3.00", None, None, None), ("L32-30", "3.00", None, None, "C30")]
CK = (("ckpt-2e+05", "0.25 M"), ("ckpt-5e+05", "0.5 M"))


def fmt(d): return f"{d['ratio']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] ({d['docs_better']}/{d['docs_worse']})"


def main():
    W = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/data/q27/w221"
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    m = json.load(open(f"{G}/kld.json")); orig = np.array([int(d["orig"]) for d in m["docs"]]); keep = ~np.isin(orig, list(BLOW))
    ctl = {s: cmp214.doc_sums(G, f"{S17}/ctl-{s}.json.rows") for s in ("2.50", "3.00")}
    def rows(lab, ck):
        for root in (f"{W}/scores", "/data/q27/w220/scores"):
            f = f"{root}/{lab}-{ck}.json.rows"
            if os.path.exists(f): return f
    def speed(lab):
        for root in (W, "/data/q27/w220"):
            f = f"{root}/runs/{lab}/train.jsonl"
            if os.path.exists(f):
                w = [r["wall"] for r in map(json.loads, open(f)) if r.get("what") == "main" and r.get("step", 0) >= 2 and "wall" in r]
                return (float(np.median(w)), len(w)) if w else None
    res = {}; md = ["| run | size | rows | vs unhealed (docs b/w) | vs control (docs b/w) | vs unhealed w/o blow-up docs | vs control w/o blow-up docs | docs > 10x (id: ctl -> arm KL/row) | w217 blow-ups (>= 0.3, >= 5x unhealed) | s / step |", "|---|---|---|---|---|---|---|---|---|---|"]
    sums = {}
    for lab, size, _, _, cl in RUNS:
        for ck, rl in CK:
            f = rows(lab, ck)
            if f: sums[(lab, ck)] = cmp214.doc_sums(G, f)[0]
    for lab, size, _, _, cl in RUNS:
        xc, n = ctl[size]; ok = n > 0
        for ck, rl in CK:
            x = sums.get((lab, ck))
            if x is None: continue
            r = {"vs_unhealed": cmp214.paired(xc, x, n)}
            cells = [fmt(r["vs_unhealed"])]
            c = sums.get((cl, ck)) if cl else None
            if cl and c is None: cells.append("(control not read)")
            elif c is None: cells.append("- (unhealed is the control)")
            else: r["vs_control"] = cmp214.paired(c, x, n); cells.append(fmt(r["vs_control"]))
            nk = n * keep; r["vs_unhealed_nobl"] = cmp214.paired(xc[keep], x[keep], n[keep]); cells.append(fmt(r["vs_unhealed_nobl"]))
            if c is not None: r["vs_control_nobl"] = cmp214.paired(c[keep], x[keep], n[keep]); cells.append(fmt(r["vs_control_nobl"]))
            else: cells.append("-")
            nn = np.maximum(n, 1); a, u = x / nn, xc / nn; ref = (c / nn) if c is not None else None
            big = ok & (n >= 8) & ((a > 10 * u) | ((ref is not None) & (a > 10 * (ref if ref is not None else u))))
            r["docs_gt10x"] = [{"doc": int(orig[i]), "gen_rows": int(n[i]), "unhealed": float(u[i]), "control": float(ref[i]) if ref is not None else None, "arm": float(a[i])} for i in np.where(big)[0]]
            cells.append("; ".join(f"{d['doc']}: {(d['control'] if d['control'] is not None else d['unhealed']):.3f} -> {d['arm']:.3f}" for d in r["docs_gt10x"]) or "none")
            bl = ok & (a >= 0.3) & (a >= 5 * u); r["blowup_docs"] = [int(orig[i]) for i in np.where(bl)[0]]; cells.append(", ".join(map(str, r["blowup_docs"])) or "none")
            sp = speed(lab) if ck == "ckpt-5e+05" else None
            cells.append(f"{sp[0]:.1f} (n={sp[1]})" if sp else "")
            if sp: r["s_per_step"] = sp[0]
            res[f"{lab}|{rl}"] = r; md.append(f"| {lab} | {size} | {rl} | " + " | ".join(cells) + " |")
    print("\n".join(md))
    for size in ("2.50", "3.00"):
        xc, n = ctl[size]; print(f"\nunhealed {size}: KL {xc.sum() / n.sum():.6f} / gen row, {int((n > 0).sum())} docs; without 1893 / 3820: {xc[keep].sum() / n[keep].sum():.6f}")
    if out: json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
