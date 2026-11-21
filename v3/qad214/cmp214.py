#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmp214.py GOLDEN CONTROL.rows ARM.rows [ARM.rows ...] [--json OUT] -- #214's paired readout: per arm, KL over the golden's gen rows vs the
control's (radiance --kld-out rows files, float32 [rows, 4], col 0 = KL), ratio arm / control with a 95 % per-document bootstrap CI
(2,000 resamples, seed 0; healeval.cmp's arithmetic), documents better / worse. Also arm vs the previous arm in the list ("step")."""
import json, sys
import numpy as np


def doc_sums(golden, rows):
    m = json.load(open(f"{golden}/kld.json")); gen = np.fromfile(f"{golden}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(rows, np.float32).reshape(-1, 4)
    assert len(K) == m["rows"], (rows, len(K), m["rows"])
    s, n = [], []
    for d in m["docs"]:
        sl = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]); g = gen[sl]
        s.append(float(K[sl, 0][g].astype(np.float64).sum())); n.append(int(g.sum()))
    return np.array(s), np.array(n)


def paired(xa, xb, n, reps=2000):
    rng = np.random.default_rng(0); bs = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n)); bs.append(xb[i].sum() / xa[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5]); ok = n > 0
    return {"ratio": float(xb.sum() / xa.sum()), "ci95": [float(lo), float(hi)], "docs_better": int(((xb < xa) & ok).sum()),
            "docs_worse": int(((xb > xa) & ok).sum())}


def main():
    a = sys.argv[1:]; out = None
    if "--json" in a: i = a.index("--json"); out = a[i + 1]; a = a[:i] + a[i + 2:]
    g, ctl, arms = a[0], a[1], a[2:]
    xc, n = doc_sums(g, ctl); rep = {"golden": g, "positions": int(n.sum()), "docs": int((n > 0).sum()), "control_kld": float(xc.sum() / n.sum()), "arms": {}}
    print(f"control {ctl}: KL {rep['control_kld']:.6f} over {int(n.sum())} gen rows, {int((n > 0).sum())} docs")
    prev, prevx = ("control", xc), None
    for r in arms:
        x, nb = doc_sums(g, r); assert (nb == n).all(), r
        d = paired(xc, x, n); d["kld"] = float(x.sum() / n.sum())
        d["step"] = paired(prev[1], x, n) if prev[0] != "control" else None
        rep["arms"][r] = d
        print(f"{r}: KL {d['kld']:.6f} ratio {d['ratio']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] better {d['docs_better']} worse {d['docs_worse']}"
              + (f" | vs previous {d['step']['ratio']:.4f} [{d['step']['ci95'][0]:.4f}, {d['step']['ci95'][1]:.4f}]" if d["step"] else ""))
        prev = (r, x)
    if out: json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
