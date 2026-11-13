#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkinit171.py MENU FIT.out.json OBS.jsonl OUTDIR SIZE [SIZE ..] -- the search's first incumbents (#171): per size the
starting fit's knapsack optimum (moves.mckp, as the driver's "solve" move) in [size - tol, size - margin] at the
driver's byte accounting (menu fixed + bytes_offset0), from the uniform level under the size; writes
OUTDIR/init-<size>.json (CONF init {size: path}) and prints predicted KLD / bpw. Sizes the menu cannot reach are
skipped. numpy only (no fit)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np    # noqa: E402
import fit as FT      # noqa: E402
import menu as MN     # noqa: E402
import moves as MV    # noqa: E402

OFF, TOL, MARGIN = int(os.environ.get("BYTES_OFFSET0", 8255936)), 0.02, 0.005


def main(menu, fitout, obs, od, sizes):
    M = MN.Menu.load(menu); rows = [json.loads(l) for l in open(obs)]; d = json.load(open(fitout))
    by = {o["tag"]: o for o in rows}; rows = [by[t] for t in d["obs_tags"]]
    Mo = FT.build(M, rows, {"beta": "0"}); Mo.layout(); pred = MV.Pred(Mo, np.array(d["x"]))
    M.fixed += OFF; os.makedirs(od, exist_ok=True); lv = M.levels()
    for s in sizes:
        hi = (s - MARGIN) * M.P / 8; lo = (s - TOL) * M.P / 8
        under = [n for n in lv if M.bpw(M.uniform(n)) <= s - MARGIN]
        if not under: print(f"{s}: below the menu's cheapest mix: skipped"); continue
        a0 = M.uniform(under[-1]); a = MV.mckp(pred, a0, hi, lo)
        if a is None or M.bytes(a) > hi: print(f"{s}: no solution in the window"); continue
        json.dump(a, open(f"{od}/init-{s:g}.json", "w"))
        print(f"{s}: predicted KLD {pred.kl(a):.5f} at {M.bpw(a):.4f} bpw (uniform {under[-1]}: {pred.kl(M.uniform(under[-1])):.5f}); head {a[MN.HEAD]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [float(x) for x in sys.argv[5:]])
