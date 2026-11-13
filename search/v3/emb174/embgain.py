#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""embgain.py FIT.in.json FIT.out.json DRVROOT [--offsets e4:-0.00059,...] [--bytes e4:B,...] [--b SIZE:b,...] -- the
search fit's embedding gain at each size's incumbent (w135, #174). The embedding term is an offset on P_0 (kfit3 /
fit.py: P_0 = c0 + emb), so a change d of it moves the predicted KLD by G d with G = dP_final / dP_0 =
prod_l (1 + kappa_l X_l) at the incumbent. Prints per size: KLD_pred, G, and for every offset given its predicted
dKLD, d log KLD, the extra bytes as bpw and the size-adjusted score change d log KLD + b dbpw (b = the driver's slope
per size); the option with the lowest score change is the predicted pick at fixed trunk. Predict only (no scipy)."""
import argparse, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import fit as FT             # noqa: E402
import menu as MN            # noqa: E402

PARAMS = 26895998464


def kv(s, f=float): return {a: f(b) for a, b in (p.split(":") for p in s.split(","))} if s else {}


def incumbents(root):
    inc = {}
    for l in open(f"{root}/state.jsonl"):
        r = json.loads(l)
        if r.get("ev") == "incumbent": inc[float(r["size"])] = r["tag"]
    return inc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="embgain.py")
    ap.add_argument("fit_in"); ap.add_argument("fit_out"); ap.add_argument("root")
    ap.add_argument("--menu", help="menu path (default: the fit input's)"); ap.add_argument("--offsets", default="")
    ap.add_argument("--bytes", default=""); ap.add_argument("--b", default=""); ap.add_argument("--json")
    a = ap.parse_args(argv)
    d = json.load(open(a.fit_in)); o = json.load(open(a.fit_out))
    M = MN.Menu.load(a.menu or d["menu"])
    Mo = FT.build(M, d["obs"], d.get("opts")); x = __import__("numpy").array(o["x"]); Mo.layout()
    tb = Mo.tables(x); off = kv(a.offsets); by = kv(a.bytes, int); bs = kv(a.b)
    out = {}
    for size, tag in sorted(incumbents(a.root).items()):
        p = f"{a.root}/assign/{tag}.json"
        if not os.path.exists(p): continue
        asg = json.load(open(p)); X = Mo.layer_X(tb, asg)
        g, P = Mo.grad_X(tb, X, Mo.D.emb_label(asg))
        G = 1.0
        for l in range(len(X)): G *= 1 + tb["kl"][l] * X[l]
        b = bs.get(f"{size:g}", 1.0)
        row = {"tag": tag, "kld_pred": P, "G": G, "b": b, "opts": {}}
        for e, dv in off.items():
            dk = G * dv; dl = math.log(max(P + dk, 1e-12) / P); db = by.get(e, 0) * 8 / PARAMS
            row["opts"][e] = {"dkld": dk, "dlog": dl, "dbpw": db, "dscore": dl + b * db}
        row["pick"] = min(row["opts"], key=lambda e: row["opts"][e]["dscore"]) if row["opts"] else None
        out[size] = row
        print(f"{size:4g} {tag:28s} KL {P:.5f} G {G:8.3f} b {b:.2f} | " +
              " ".join(f"{e} {r['dlog']:+.4f}/{r['dscore']:+.4f}" for e, r in row["opts"].items()) + f" | pick {row['pick']}")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
