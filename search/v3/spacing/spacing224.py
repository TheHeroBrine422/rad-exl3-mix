# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""spacing224.py CODE_V3 MENU OUT.json BAND=IN.json:FIT.out.json:SIZES ... -- #224: what the K-grid spacing of the trunk costs
(predicted KL of the model's knapsack optimum at each size), K steps 1, 1/2, 1/4, 1/8, 1/16 and "continuous" (1/64), vs the menu as
measured (1/8 to K5, 1/4 to K7, K7.5).

Model: the V3 search's fitted kfit3 / fit202 model of the band (IN = the fit's input, FIT = its output; the srch183 fits on colo).
The model is built on the measured menu (its per-tensor error normalisers eb fixed), then predicts on an AUGMENTED menu: every
trunk tensor (blk.*, family v3mad) gets options on the spacing's grid K = n s inside its measured range [Kmin, Kmax]; a grid point
that is a measured level keeps its measured bytes / error; one between measured levels is interpolated: bytes linear in K,
log(error) linear in K (an assumption for every off-grid level: 1/16 everywhere, 1/8 above K5, 1/64). Head (output.weight) and
embedding keep their measured options. Solver: moves.mckp (Lagrangian MCKP on costs linearised at the current solution,
re-linearised; exact compounding for the reported KL), window [size - 0.02, size] bpw, started from the measured-menu optimum
(re-run to convergence: iters 8). Lagrangian solutions lie on each tensor's convex hull, the same solver as the releases.
Prints / writes per band and size: predicted KL, bytes (bpw), ratio vs continuous, ratio vs the measured menu."""
import copy, json, math, sys

import numpy as np

SP = [("1", 1.0), ("1/2", 0.5), ("1/4", 0.25), ("1/8", 0.125), ("1/16", 0.0625), ("cont(1/64)", 1 / 64)]


def augment(menu, s):
    m = copy.deepcopy(menu)
    for t in m["tensors"]:
        if not t["w"].startswith("blk."): continue
        oo = sorted([o for o in t["opts"] if o.get("fam") == "v3mad"], key=lambda o: o["K"])
        other = [o for o in t["opts"] if o.get("fam") != "v3mad"]
        K = np.array([o["K"] for o in oo]); B = np.array([o["bytes"] for o in oo], float); LE = np.log([o["err"] for o in oo])
        byK = {round(o["K"], 6): o for o in oo}
        n0, n1 = math.ceil(K[0] / s - 1e-9), math.floor(K[-1] / s + 1e-9)
        new = []
        for n in range(n0, n1 + 1):
            k = round(n * s, 6)
            if k in byK: new.append(byK[k]); continue
            new.append({"o": f"{t['w']}@K{k}", "K": k, "bytes": float(np.interp(k, K, B)), "err": float(np.exp(np.interp(k, K, LE))), "fam": "v3mad"})
        t["opts"] = new + other
    return m


def main():
    code, menup, outp = sys.argv[1:4]; sys.path.insert(0, code)
    import fit as F2, menu as MN, moves as MV   # srch183 fits are fit.py (no floor term)
    menu = json.load(open(menup)); res = []
    for spec in sys.argv[4:]:
        inp, fitp, sizes = spec.split(":"); sizes = [float(z) for z in sizes.split(",")]
        d = json.load(open(inp)); ft = json.load(open(fitp))
        M0 = MN.Menu(menu); Mo = F2.build(M0, d["obs"], d["opts"]); (Mo.layout() if hasattr(Mo, "layout") else None); x = np.array(ft["x"]); assert len(x) == Mo.N
        P0 = MV.Pred(Mo, x)
        a_mid = {t["w"]: t["opts"][len(t["opts"]) // 2]["o"] for t in M0.T}
        for size in sizes:
            hi = size * M0.P / 8; lo = (size - 0.02) * M0.P / 8
            a0 = MV.mckp(P0, a_mid, hi, lo, iters=8); k0 = P0.kl(a0)
            row = {"band": inp.split("/")[-1], "size": size, "menu": {"kl": k0, "bpw": M0.bpw(a0)}}
            for name, s in SP:
                Ma = MN.Menu(augment(menu, s)); Mo.D.M = Ma; P = MV.Pred(Mo, x)
                start = {w: (o if w not in Ma.WI or o in Ma.oi[Ma.WI[w]] else min(Ma.T[Ma.WI[w]]["opts"], key=lambda q: abs(q["K"] - M0.opt(w, o)["K"]))["o"]) for w, o in a0.items()}
                a = MV.mckp(P, start, hi, lo, iters=8); row[name] = {"kl": P.kl(a), "bpw": Ma.bpw(a),
                    "Kmean": float(np.mean([Ma.opt(w, a[w])["K"] for w in a if w.startswith("blk.")]))}
                Mo.D.M = M0
            c = row["cont(1/64)"]["kl"]
            print(f"{row['band']} {size:.2f}  menu {k0 / c:.4f}  " + "  ".join(f"{n} {row[n]['kl'] / c:.4f} ({row[n]['bpw']:.3f})" for n, _ in SP), flush=True)
            res.append(row)
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
