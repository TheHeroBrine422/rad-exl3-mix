#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmp224.py CONTROL/eval.json ARM/eval.json [ARM/eval.json ...] [--json OUT] -- w224's paired readout of qad220's --eval-only passes (in-trainer
KL, not the served read's): per arm, KL over the golden's gen rows, ratio arm / control with a 95 % per-document bootstrap CI (cmp214.paired's
arithmetic: 2,000 resamples, seed 0), documents better / worse. All files must be passes over the same golden (same documents, same rows)."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qad214"))
from cmp214 import paired  # noqa: E402


def load(f):
    z = json.load(open(f)); return z, np.array([d["kl_sum"] for d in z["docs"]]), np.array([d["rows"] for d in z["docs"]])


def main():
    a = sys.argv[1:]; out = None
    if "--json" in a: i = a.index("--json"); out = a[i + 1]; a = a[:i] + a[i + 2:]
    zc, xc, n = load(a[0]); rep = {"golden": zc["golden"], "control": a[0], "control_kl": float(xc.sum() / n.sum()), "rows": int(n.sum()), "arms": {}}
    print(f"control {a[0]}: KL {rep['control_kl']:.6f} over {int(n.sum())} gen rows, {int((n > 0).sum())} docs ({zc['golden']})")
    for f in a[1:]:
        z, x, nb = load(f); assert z["golden"] == zc["golden"] and (nb == n).all(), f"{f}: not the control's golden / rows"
        d = paired(xc, x, n); d["kl"] = float(x.sum() / n.sum()); rep["arms"][f] = d
        print(f"{f}: KL {d['kl']:.6f} ratio {d['ratio']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] better {d['docs_better']} worse {d['docs_worse']}")
    if out: json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
