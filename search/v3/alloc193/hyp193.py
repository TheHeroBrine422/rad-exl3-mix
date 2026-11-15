# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""hyp193.py SNAP OUT.json -- offline read of the amendment-7 hypotheses on the allowed paired outcomes (pairs193):
does a candidate's change of roughness / contrast / top-decile K / bottom-decile K versus its reference predict the
measured stage-1 log r? Weighted least squares per band (weights 1 / (se^2 + 0.005^2), size intercepts), features
standardized within band, one feature at a time and jointly; SE by bootstrap over pairs (300). Report only: written
after the round-1 files of amendment 7 were generated; not a gate (low bar)."""
import json, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common193 as C  # noqa: E402
import pairs193 as P  # noqa: E402
import menu as MN  # noqa: E402

FE = ["d_rough", "d_contrast", "d_top", "d_bot"]


def main(snap, out):
    M = C.load_menu(snap); km = C.KM(M); rows = [r for r in P.build(snap) if r["src"] == "stage1" and r["size"] in C.SIZES]
    W = [w for w in km.W if w not in (MN.HEAD, MN.EMB)]
    nper = {}
    for t in M.T:
        o = sorted(t["opts"], key=lambda x: x["bytes"]); nper[t["w"]] = (o[-1]["bytes"] - o[0]["bytes"]) / max(o[-1]["K"] - o[0]["K"], 1e-9)
    nv = np.array([nper[w] for w in W]); nv = nv / nv.sum()
    pos = {}
    for i, w in enumerate(W):
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", w); pos[(m.group(2), int(m.group(1)))] = i
    nb = [[pos[(t, l + d)] for d in (-1, 1) if (t, l + d) in pos] for (t, l), i in sorted(pos.items(), key=lambda kv: kv[1])]
    cache = {}

    def K(tag):
        if tag not in cache:
            a = C.assign(snap, tag); cache[tag] = np.array([km.Kof[w][a[w]] for w in W])
        return cache[tag]

    def rough(k): return float(sum(nv[i] * (k[i] - np.mean(k[n])) ** 2 for i, n in enumerate(nb) if n))

    X, y, wt, band, size = [], [], [], [], []
    for r in rows:
        try: kc, kr = K(r["cand"]), K(r["refa"])
        except (FileNotFoundError, KeyError): continue
        m = float((kr * nv).sum()); top = kr >= np.percentile(kr, 90); bot = kr <= np.percentile(kr, 10)
        X.append([rough(kc) - rough(kr), float((nv * (kc - (kc * nv).sum()) ** 2).sum() - (nv * (kr - m) ** 2).sum()),
                  float((kc - kr)[top].mean()), float((kc - kr)[bot].mean())])
        y.append(r["logr"]); wt.append(1 / (r["se"] ** 2 + 0.005 ** 2)); band.append(C.BAND[r["size"]]); size.append(r["size"])
    X, y, wt, band, size = np.array(X), np.array(y), np.array(wt), np.array(band), np.array(size)
    rng = np.random.default_rng(193); res = {"n": len(y), "features": FE, "bands": {}}

    def fit(Xb, yb, wb, sb, cols):
        S = sorted(set(sb)); A = np.column_stack([Xb[:, cols]] + [(sb == s).astype(float) for s in S]); sw = np.sqrt(wb)
        return np.linalg.lstsq(A * sw[:, None], yb * sw, rcond=None)[0][:len(cols)]
    for b in ("b1", "b2", "b3", "all"):
        m = np.ones(len(y), bool) if b == "all" else band == b
        Xb = X[m] / X[m].std(0); yb, wb, sb = y[m], wt[m], size[m]; o = {"n": int(m.sum())}
        for name, cols in [(f, [i]) for i, f in enumerate(FE)] + [("joint", list(range(len(FE))))]:
            est = fit(Xb, yb, wb, sb, cols); bs = []
            for _ in range(300):
                ix = rng.integers(0, len(yb), len(yb)); bs.append(fit(Xb[ix], yb[ix], wb[ix], sb[ix], cols))
            se = np.std(bs, 0); o[name] = [[float(e), float(s), float(e / s)] for e, s in zip(est, se)]
        res["bands"][b] = o
        print(f"{b} (n {o['n']}): " + "; ".join(f"{f} {o[f][0][0]:+.4f} (z {o[f][0][2]:+.1f})" for f in FE) + " | joint z " + " ".join(f"{v[2]:+.1f}" for v in o["joint"]))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
