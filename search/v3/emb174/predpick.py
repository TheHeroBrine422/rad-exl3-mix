#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""predpick.py FIT.in.json FIT.out.json INCUMBENTS.json [--tol 0.02] [--json OUT] -- the predicted embedding choice per
size (w135, #174): for each size's incumbent and every embedding option, the driver's own model optimum at that size
(moves._mckp1: trunk + head re-optimised under the byte window [size - tol, size] incl. the driver's byte offset,
embedding fixed) and its predicted
KLD; the pick = the lowest (moves.mckp's rule). INCUMBENTS: {size: assignment file}. CPU only (run on colo)."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import fit as FT             # noqa: E402
import menu as MN            # noqa: E402
import moves as MV           # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="predpick.py")
    ap.add_argument("fit_in"); ap.add_argument("fit_out"); ap.add_argument("inc")
    ap.add_argument("--tol", type=float, default=0.02); ap.add_argument("--json"); ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--bytes-off", type=int, default=8255936, help="the driver's container - menu byte offset (CONF bytes_offset0)")
    a = ap.parse_args(argv)
    d = json.load(open(a.fit_in)); o = json.load(open(a.fit_out))
    M = MN.Menu.load(d["menu"]); M.fixed += a.bytes_off; Mo = FT.build(M, d["obs"], d.get("opts")); Mo.layout(); x = np.array(o["x"])
    pred = MV.Pred(Mo, x); embs = [e["o"] for e in M.T[M.WI[MN.EMB]]["opts"]]
    print("fitted embedding offsets", {k: round(v, 6) for k, v in Mo.tables(x)["emb"].items()}, "c0", round(Mo.tables(x)["c0"], 6))
    out = {}
    for size, p in sorted(json.load(open(a.inc)).items(), key=lambda kv: float(kv[0])):
        inc = json.load(open(p)); s = float(size)
        hi = s * M.P / 8; lo = (s - a.tol) * M.P / 8
        row = {"inc_emb": inc.get(MN.EMB), "inc_pred": pred.kl(inc), "opts": {}}
        for e in embs:
            cur = MV._mckp1(pred, dict(inc, **{MN.EMB: e}), hi, lo, a.iters)
            if cur is None: continue
            row["opts"][e] = {"pred": pred.kl(cur), "bpw": M.bpw(cur), "in": bool(lo <= M.bytes(cur) <= hi)}
        ok = {e: r for e, r in row["opts"].items() if r["in"]}
        row["pick"] = min(ok, key=lambda e: ok[e]["pred"]) if ok else None
        base = row["opts"].get("base", {}).get("pred")
        out[size] = row
        print(f"{s:4g} inc {row['inc_emb']} {row['inc_pred']:.5f} | " + " ".join(
            f"{e} {r['pred']:.5f}{'' if r['in'] else '*'}" + (f" ({np.log(r['pred'] / base):+.3f})" if base else "") for e, r in row["opts"].items())
            + f" | pick {row['pick']}")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
