#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""parts131.py SPECS RESULTS DOCSDIR MENU OUT.json -- #171 test A (search/v3/RULES-parts.md): per-part labels and the
optional x ratio scale (fit.py opts xscale) on w129's rows. β 0; train = the fit129 U + G rows + the s129 anchors
u-k2.5 / u-k4; held out = the 8 s129 H swaps, each predicted as its log ratio to its base on the same label.
Arms: A pooled s129 label; C parts as own labels (s129|x, s129|c); E = C + xscale. Runs on colo (scipy, ~1 min/arm)."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w129")); sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import fit as FT      # noqa: E402
import fit129 as F9   # noqa: E402
import menu as MN     # noqa: E402

ARMS = (("A", False, False), ("C", True, False), ("E", True, True))


def main(specs, results, docsdir, menu, out):
    S, R, _ = F9.load_rows(specs, results, docsdir); M = MN.Menu.load(menu)
    a = lambda t: F9.assign_of(S[t])
    fitrows = [{"tag": f"{t}@fit129", "assign": a(t), "label": "fit129", "kld": r["kld"], "se": 0.0}
               for (t, sl), r in R.items() if sl == "fit129" and S[t]["w129"]["kind"] in ("U", "G")]
    H = sorted(t for (t, sl) in R if sl == "s129" and S[t]["w129"]["kind"] == "G")
    rep = {}
    for arm, parts, xs in ARMS:
        rows, held = list(fitrows), []
        for t in ["w129-u-k2.5", "w129-u-k4"] + H:
            r = R[(t, "s129")]
            labs = [("s129|x", r["parts"]["x"]["kld"]), ("s129|c", r["parts"]["c"]["kld"])] if parts else [("s129", r["kld"])]
            for lab, k in labs:
                (held if t in H else rows).append({"tag": f"{t}@{lab}", "assign": a(t), "label": lab, "kld": k, "se": 0.0})
        Mo = FT.build(M, rows + held, {"beta": "0", "xscale": xs}); n = len(rows)
        x = Mo.fit(idx=list(range(n)))
        P = Mo.predict_rows(x, list(range(len(rows) + len(held))))
        by = {o["tag"]: (P[i], o["kld"]) for i, o in enumerate(rows + held)}
        errs = {}
        for o in held:
            t, lab = o["tag"].split("@"); bt = f"w129-u-k{S[t]['w129']['base']:g}@{lab}"
            (pr, me), (pb, mb) = by[o["tag"]], by[bt]
            errs[o["tag"]] = {"pred": math.log(pr / pb), "meas": math.log(me / mb)}
        rm = lambda ks: float(np.sqrt(np.mean([(errs[k]["pred"] - errs[k]["meas"]) ** 2 for k in ks]))) if ks else None
        rep[arm] = {"parts": parts, "xscale": xs, "lamx": Mo.lam(x), "rmse_all": rm(list(errs)),
                    "rmse_x": rm([k for k in errs if k.endswith("|x")]), "rmse_c": rm([k for k in errs if k.endswith("|c")]),
                    "train_rmse_log": float(np.sqrt(np.mean(np.log(P[:n] / np.array([o["kld"] for o in rows])) ** 2))), "rows": errs}
        print(f"[{arm}] parts {parts} xscale {xs}: held-out log-RMSE all {rep[arm]['rmse_all']:.4f} x {rep[arm]['rmse_x']} c {rep[arm]['rmse_c']} "
              f"lambda {rep[arm]['lamx']:.3f} (train {rep[arm]['train_rmse_log']:.4f})", flush=True)
    C, E = rep["C"], rep["E"]
    adopt = E["rmse_x"] <= 0.9 * C["rmse_x"] and E["rmse_c"] <= C["rmse_c"] + 0.005
    rep["verdict"] = {"adopt_xscale": adopt, "rule": "E x <= 0.9 C x and E c <= C c + 0.005 (RULES-parts.md test A)"}
    print(f"verdict: adopt xscale = {adopt}")
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
