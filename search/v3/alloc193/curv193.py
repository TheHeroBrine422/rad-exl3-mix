# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""curv193.py SNAP REFITDIR OUT.json [--pairs PAIRS.jsonl --pred-out PRED.json] -- from my fit.py refits on the allowed
rows (RULES-193 §0; REFITDIR/refit-b{1,2,3}.{in,out}.json):
  - per size (1.25 ... 6.0) at the size's current incumbent, per weight: K, the curvature c = 1/2 d2 log KL / dK2 and the
    slope g = d log KL / dK (3-point differences over the weight's own-family menu neighbours, one-sided at the ends);
  - fit.py's sensitivity-optimal allocation at each size (moves.mckp from the incumbent, the size's window): `opt`;
  - (--pairs) the refit's predicted log r of every paired row (RULES-193 §3 version ii).
Bands: b1 = sizes <= 2.5, b2 = 3-4, b3 = 4.5-6 (the driver's fit_groups)."""
import argparse, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import common193 as C  # noqa: E402
import fit as FT  # noqa: E402
import moves as MV  # noqa: E402

ALLS = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def band(s): return "b1" if s <= 2.5 + 1e-9 else ("b2" if s <= 4.0 + 1e-9 else "b3")


def load_pred(M, rdir, g):
    d = json.load(open(f"{rdir}/refit-{g}.in.json")); o = json.load(open(f"{rdir}/refit-{g}.out.json"))
    Mo = FT.build(M, d["obs"], d.get("opts", {}))
    return MV.Pred(Mo, np.array(o["x"])), o["report"]


def curv(pred, km, a):
    out = {}
    base = math.log(pred.kl(a))
    for w in km.W:
        opts = km.fam_opts(w, km.fam(a[w])); names = [o[1] for o in opts]; j = names.index(a[w])
        idx = [j - 1, j, j + 1] if 0 < j < len(opts) - 1 else ([0, 1, 2] if j == 0 else [j - 2, j - 1, j])
        idx = [i for i in idx if 0 <= i < len(opts)]
        if len(idx) < 3: out[w] = [opts[j][0], 1e-6, 0.0]; continue
        ks = [opts[i][0] for i in idx]; ls = []
        for i in idx:
            if i == j: ls.append(base); continue
            b = dict(a); b[w] = opts[i][1]; ls.append(math.log(pred.kl(b)))
        (x0, x1, x2), (y0, y1, y2) = ks, ls
        d1 = (y1 - y0) / (x1 - x0); d2 = (y2 - y1) / (x2 - x1)
        c = (d2 - d1) / (x2 - x0)                                   # = 1/2 second derivative
        g = d1 + c * (2 * opts[j][0] - x0 - x1)                       # slope of the parabola at the current K
        out[w] = [opts[j][0], float(max(c, 1e-6)), float(g)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("rdir"); ap.add_argument("out")
    ap.add_argument("--pairs"); ap.add_argument("--pred-out"); ap.add_argument("--no-opt", action="store_true")
    a = ap.parse_args()
    M = C.load_menu(a.snap); km = C.KM(M); st = C.load_state(a.snap)
    inc = {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
    P = {}; rep = {}
    for g in ("b1", "b2", "b3"): P[g], rep[g] = load_pred(M, a.rdir, g); print(g, rep[g], flush=True)
    res = {"report": rep, "inc": {}, "curv": {}, "opt": {}}
    for s in ALLS:
        ia = C.assign(a.snap, inc[s]); pr = P[band(s)]
        res["inc"][f"{s:g}"] = inc[s]; res["curv"][f"{s:g}"] = curv(pr, km, ia)
        if not a.no_opt:
            lo, hi = C.window(M, s); oa = MV.mckp(pr, ia, hi, lo)
            res["opt"][f"{s:g}"] = oa
            print(f"{s:g}: opt pred log r vs inc {math.log(pr.kl(oa) / pr.kl(ia)):+.4f}, bpw {M.bpw(oa):.4f}, "
                  f"{sum(oa[w] != ia[w] for w in km.W)} tensors differ", flush=True)
    json.dump(res, open(a.out, "w"))
    if a.pairs:
        pd = {}
        for l in open(a.pairs):
            r = json.loads(l); pr = P[band(r["size"])]
            pd[r["tag"]] = math.log(pr.kl(C.assign(a.snap, r["cand"])) / pr.kl(C.assign(a.snap, r["refa"])))
        json.dump(pd, open(a.pred_out, "w")); print("pred rows", len(pd))


if __name__ == "__main__":
    main()
