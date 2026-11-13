# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""abreport.py DIR [--out OUT] [--png PNG] -- the #167 A/B report (w127; rule RULES-sizeadj.md): raw-KLD vs size-adjusted
accept on the CPU stand-in. DIR holds s<seed>-raw.traj / s<seed>-adj.traj (standin.py --traj).

Per (seed, size): the incumbent's true size-adjusted score S = log KL_true + b_true (bpw_container - s), gap = S -
log KL*(s) (the truth's knapsack optimum at s); final dS = S_adj - S_raw; convergence = the gap on an eval grid and on
a card-hour grid (0 -> the pair's shorter end, 40 points), dAUC = mean of the per-point gap difference; CIs by a seed
bootstrap (10,000 draws; a seed's value = the mean over its sizes). Also: bpw drift (bpw_container - s), accepts and
wrong accepts (true dS > +0.5% at the accept; raw-wrong: true d log KL > +0.5%), fitted b vs b_true."""
import argparse, glob, json, math, os, re, sys

import numpy as np


def load(p):
    T = None; inc = []; acc = []; end = None
    for l in open(p):
        e = json.loads(l)
        if e["ev"] == "truth": T = e
        elif e["ev"] == "inc": inc.append(e)
        elif e["ev"] == "accept": acc.append(e)
        elif e["ev"] == "end": end = e
    return T, inc, acc, end


def curve(inc, size, F, key):
    """step function of (x, gap, bpw drift, raw log KL) for one size; x = evals ("n") or card-hours"""
    out = []
    for e in inc:
        if e["size"] != size: continue
        S = math.log(e["kl"]) + F["b_true"] * (e["bpw"] - size)
        out.append((e[key], S - F["logkl_opt"], e["bpw"] - size, math.log(e["kl"])))
    return out


def at(c, x):
    """the value of a step curve at x (the last point with x_i <= x; the first point before it starts)"""
    v = c[0]
    for p in c:
        if p[0] <= x: v = p
        else: break
    return v


def boot(vals_by_seed, n=10000, seed=0):
    s = sorted(vals_by_seed); v = np.array([vals_by_seed[k] for k in s]); rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="abreport.py"); ap.add_argument("dir"); ap.add_argument("--out"); ap.add_argument("--png")
    a = ap.parse_args(argv)
    runs = {}
    for p in glob.glob(f"{a.dir}/s*-*.traj"):
        m = re.match(r"s(\d+)-(raw|adj)\.traj$", os.path.basename(p))
        if m: runs[(int(m.group(1)), m.group(2))] = load(p)
    seeds = sorted(s for s in {k[0] for k in runs} if (s, "raw") in runs and (s, "adj") in runs and runs[(s, "raw")][3] and runs[(s, "adj")][3])
    if not seeds: raise SystemExit("no finished pairs")
    sizes = sorted(float(z) for z in runs[(seeds[0], "raw")][0]["frontier"])
    R = {"seeds": seeds, "sizes": sizes, "cells": [], "arms": {}}
    G = 40
    curves = {arm: {z: {"n": [], "h": []} for z in sizes} for arm in ("raw", "adj")}
    dfin, dauc_h, dauc_n = {}, {}, {}
    for s in seeds:
        dfin[s] = []; dauc_h[s] = []; dauc_n[s] = []
        for z in sizes:
            F = runs[(s, "raw")][0]["frontier"][str(z)]
            cn = {arm: curve(runs[(s, arm)][1], z, F, "n") for arm in ("raw", "adj")}
            ch = {arm: curve(runs[(s, arm)][1], z, F, "card_h") for arm in ("raw", "adj")}
            nend = min(runs[(s, arm)][3]["evals"] for arm in ("raw", "adj")); hend = min(runs[(s, arm)][3]["card_h"] for arm in ("raw", "adj"))
            gn = np.linspace(0, nend, G + 1)[1:]; gh = np.linspace(0, hend, G + 1)[1:]
            for arm in ("raw", "adj"):
                curves[arm][z]["n"].append([at(cn[arm], x)[1] for x in gn]); curves[arm][z]["h"].append([at(ch[arm], x / 1.0)[1] for x in gh])
            fr = {arm: at(cn[arm], nend) for arm in ("raw", "adj")}
            d = fr["adj"][1] - fr["raw"][1]
            ah = float(np.mean([at(ch["adj"], x)[1] - at(ch["raw"], x)[1] for x in gh]))
            an = float(np.mean([at(cn["adj"], x)[1] - at(cn["raw"], x)[1] for x in gn]))
            dfin[s].append(d); dauc_h[s].append(ah); dauc_n[s].append(an)
            R["cells"].append({"seed": s, "size": z, "b_true": F["b_true"], "gap_raw": fr["raw"][1], "gap_adj": fr["adj"][1], "dS": d,
                               "dauc_h": ah, "dauc_n": an, "drift_raw": fr["raw"][2], "drift_adj": fr["adj"][2],
                               "kl_raw": math.exp(fr["raw"][3]), "kl_adj": math.exp(fr["adj"][3]),
                               "kl_nominal_raw": math.exp(fr["raw"][1] + F["logkl_opt"]), "kl_nominal_adj": math.exp(fr["adj"][1] + F["logkl_opt"]),
                               "b_fit_adj": runs[(s, "adj")][3]["b"][str(z)]})
    R["primary_dS"] = boot({s: float(np.mean(v)) for s, v in dfin.items()})
    R["dAUC_card_h"] = boot({s: float(np.mean(v)) for s, v in dauc_h.items()})
    R["dAUC_evals"] = boot({s: float(np.mean(v)) for s, v in dauc_n.items()})
    R["per_size"] = {str(z): {"dS": boot({c["seed"]: c["dS"] for c in R["cells"] if c["size"] == z}),
                              "drift_raw": float(np.mean([c["drift_raw"] for c in R["cells"] if c["size"] == z])),
                              "drift_adj": float(np.mean([c["drift_adj"] for c in R["cells"] if c["size"] == z])),
                              "gap_raw": float(np.mean([c["gap_raw"] for c in R["cells"] if c["size"] == z])),
                              "gap_adj": float(np.mean([c["gap_adj"] for c in R["cells"] if c["size"] == z])),
                              "b_true": float(np.mean([c["b_true"] for c in R["cells"] if c["size"] == z])),
                              "b_fit": float(np.mean([c["b_fit_adj"] for c in R["cells"] if c["size"] == z]))} for z in sizes}
    for arm in ("raw", "adj"):
        acc = wrong = wrong_raw = 0; ev = ch = stp = rej = 0
        for s in seeds:
            T, inc, A, end = runs[(s, arm)]
            for e in A:
                F = T["frontier"][str(e["size"])]; acc += 1
                wrong += e["dlogkl"] + F["b_true"] * e["dbpw"] > 0.005; wrong_raw += e["dlogkl"] > 0.005
            ev += end["evals"]; ch += end["card_h"]; stp += end["stopped"]; rej += end["rejects"]
        R["arms"][arm] = {"accepts": acc, "wrong_adj": int(wrong), "wrong_raw": int(wrong_raw), "evals": ev, "card_h": ch, "rejects": rej, "stopped": stp,
                          "pass": [runs[(s, arm)][3]["pass"] for s in seeds]}
    wa, wr = R["arms"]["adj"]["wrong_adj"], R["arms"]["raw"]["wrong_adj"]
    worse = {"a": R["primary_dS"][0] > 0 and R["primary_dS"][1][0] > 0, "b": R["dAUC_card_h"][0] > 0 and R["dAUC_card_h"][1][0] > 0, "c": wa > 2 * wr + 2}
    R["worse"] = worse; R["decision"] = "keep raw (size-adjusted is worse)" if any(worse.values()) else "ADOPT size-adjusted"
    for arm in ("raw", "adj"):                                         # mean curves for the plot
        for z in sizes:
            for k in ("n", "h"): curves[arm][z][k] = np.mean(curves[arm][z][k], 0).tolist()
    R["curves"] = curves
    f = lambda m: f"{m[0]:+.4f} [{m[1][0]:+.4f}, {m[1][1]:+.4f}]"
    print(f"seeds {seeds}, sizes {sizes}")
    print(f"primary final dS (adj - raw, true size-adjusted log KL): {f(R['primary_dS'])}")
    print(f"dAUC vs card-hours: {f(R['dAUC_card_h'])}; vs evals: {f(R['dAUC_evals'])}")
    print(f"{'size':>5} {'b_true':>6} {'b_fit':>6} {'gap raw':>8} {'gap adj':>8} {'dS [95% CI]':>28} {'drift raw':>9} {'drift adj':>9}")
    for z in sizes:
        p = R["per_size"][str(z)]
        print(f"{z:5g} {p['b_true']:6.3f} {p['b_fit']:6.3f} {p['gap_raw']:8.4f} {p['gap_adj']:8.4f} {f(p['dS']):>28} {p['drift_raw']:+9.4f} {p['drift_adj']:+9.4f}")
    for arm, v in R["arms"].items():
        print(f"{arm}: {v['evals']} evals, {v['card_h']:.1f} card-h, accepts {v['accepts']}, wrong (true dS > +0.5%) {v['wrong_adj']}, "
              f"raw-wrong (d log KL > +0.5%) {v['wrong_raw']}, rejects {v['rejects']} ({v['stopped']} stopped early), checks pass {v['pass']}")
    print(f"worse: {worse} -> {R['decision']}")
    if a.out: json.dump(R, open(a.out, "w"), indent=1)
    if a.png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, len(sizes), figsize=(4.2 * len(sizes), 7.5), squeeze=False)
        for j, z in enumerate(sizes):
            for arm, col in (("raw", "tab:red"), ("adj", "tab:blue")):
                y = np.array(curves[arm][z]["h"]); ax[0][j].plot(np.linspace(0, 1, G + 1)[1:], y, color=col, label=arm)
                y = np.array(curves[arm][z]["n"]); ax[1][j].plot(np.linspace(0, 1, G + 1)[1:], y, color=col, label=arm)
            ax[0][j].set_title(f"{z:g} bpw: true size-adjusted gap to the optimum"); ax[0][j].set_xlabel("fraction of the pair's card-hours")
            ax[1][j].set_xlabel("fraction of the evals"); ax[0][j].legend(); ax[0][j].grid(alpha=0.3); ax[1][j].grid(alpha=0.3)
        ax[0][0].set_ylabel("gap (log KL units), mean over seeds"); ax[1][0].set_ylabel("gap vs evals")
        fig.tight_layout(); fig.savefig(a.png, dpi=90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
