# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""extra_solve.py IN.json OUT.json STATE.json SIZES NTS OUT_ASSIGNS.json -- RULES-picks am. 11-16 6:15 PM (a): knapsack
optimum + NTS Thompson draws from a driver fit (fit.py output) at SIZES, from each size's final incumbent (STATE: solve202.py
state), window [size - 0.02, size]; only allocations new vs the incumbent and every submitted hash. (colo: fit.py / moves.py)"""
import json, math, random, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refit"))
import fit as FT, menu as MN, moves as MV
inp, fitp, stp, sizes, nts, outp = sys.argv[1:7]; nts = int(nts)
d = json.load(open(inp)); ft = json.load(open(fitp)); S = json.load(open(stp))
M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
Mo = FT.build(M, d["obs"], d.get("opts") or {}); Mo.layout()
import numpy as np
pred = MV.Pred(Mo, np.array(ft["x"])); draws = [MV.Pred(Mo, np.array(v)) for v in ft["draws"]]
seen = set(S["seen"]); out = []
for size in [float(z) for z in sizes.split(",")]:
    a0 = S["inc"][str(size)]["assign"]; hi = size * M.P / 8; lo = (size - 0.02) * M.P / 8; got = 0
    order = list(range(len(draws))); random.Random(203 + int(size * 100)).shuffle(order)
    for var, p in [("opt", pred)] + [(f"ts{j}", draws[j]) for j in order]:
        if var != "opt" and got >= nts: break
        c = MV.mckp(p, a0, hi, lo)
        if c is None: continue
        h = MV.ahash(c)
        if h in seen or h == MV.ahash(a0): continue
        seen.add(h); got += var != "opt"
        out.append({"size": size, "var": var, "hash": h, "assign": c, "bpw": M.bpw(c), "pred_logr": math.log(pred.kl(c) / pred.kl(a0)),
                    "changed": sum(1 for w in c if c[w] != a0[w])})
        print(size, var, round(M.bpw(c), 4), out[-1]["changed"], round(out[-1]["pred_logr"], 4), flush=True)
json.dump(out, open(outp, "w"))
