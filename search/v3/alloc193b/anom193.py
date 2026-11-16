# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""anom193.py SNAP PAIRS.jsonl OUT.json [--pred PRED.json] -- RULES-193 §3 anomaly finder.

Groups g = 9 types x 8 depth bands + head + emb. x_jg = sum_{t in g} n_t (K_cand,t - K_ref,t) / sum_t n_t (bpw moved into
g; n_t = the tensor's bytes per K step x 8). Weighted ridge, weights 1 / (se^2 + 0.005^2): y_j = a_size + sum_g
(beta_g + delta_{g,band}) x_jg, ridge lam on beta, lam_dev on delta (chosen by 5-fold CV grouped by candidate). Bytes
are (nearly) fixed per size, so beta is identified only relative to the average group: read beta_g as "a bpw moved into g
from the average group changes log KL by beta_g".
Versions: (i) y = measured log r; (ii) y = measured - fit.py refit's predicted log r (PRED.json: tag -> pred, made by
predref193.py). SE by bootstrap over candidates (500, seed 193)."""
import argparse, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common193 as C  # noqa: E402

BANDS = ["b1", "b2", "b3"]


def groups(km):
    G = sorted({C.gname(w) for w in km.W}, key=lambda g: (g in ("head", "emb"), g))
    return G, {w: G.index(C.gname(w)) for w in km.W}


def nper(M, km):
    """bytes per unit K (x 8 = params) per weight, from the menu's mad / family slope"""
    out = {}
    for t in M.T:
        o = sorted(t["opts"], key=lambda x: x["bytes"]); a, b = o[0], o[-1]
        out[t["w"]] = 8 * (b["bytes"] - a["bytes"]) / max(b["K"] - a["K"], 1e-9)
    return out


def design(rows, snap, km, M):
    G, gi = groups(km); n = nper(M, km); tot = sum(n.values())
    cache = {}
    def kv(tag):
        if tag not in cache: a = C.assign(snap, tag); cache[tag] = np.array([km.Kof[w][a[w]] for w in km.W])
        return cache[tag]
    nv = np.array([n[w] for w in km.W]); g = np.array([gi[w] for w in km.W])
    X = np.zeros((len(rows), len(G)))
    for j, r in enumerate(rows):
        d = (kv(r["cand"]) - kv(r["refa"])) * nv / tot
        np.add.at(X[j], g, d)
    return G, X


def fit(X, B, S, y, w, lam, lamd):
    """ridge: columns = size intercepts (unpenalised) | beta (lam) | band deviations (lamd)"""
    nS = S.shape[1]; nG = X.shape[1]
    Z = np.hstack([S, X] + [X * (B == b)[:, None] for b in range(len(BANDS))])
    pen = np.r_[np.zeros(nS), np.full(nG, lam), np.full(nG * len(BANDS), lamd)]
    sw = np.sqrt(w)
    A = (Z * sw[:, None]); coef = np.linalg.solve(A.T @ A + np.diag(pen), A.T @ (y * sw))
    return coef, Z


def cv(X, B, S, y, w, grp, lam, lamd, k=5, seed=193):
    rng = np.random.default_rng(seed); ug = np.unique(grp); fold = dict(zip(ug, rng.integers(0, k, len(ug))))
    f = np.array([fold[x] for x in grp]); err = 0.0
    for i in range(k):
        tr = f != i; te = ~tr
        coef, _ = fit(X[tr], B[tr], S[tr], y[tr], w[tr], lam, lamd)
        _, Z = fit(X[te], B[te], S[te], y[te], w[te], lam, lamd) if te.sum() > 400 else (None, None)
        Zt = np.hstack([S[te], X[te]] + [X[te] * (B[te] == b)[:, None] for b in range(len(BANDS))])
        err += float(np.sum(w[te] * (y[te] - Zt @ coef) ** 2))
    return err / w.sum()


def run(rows, X, G, y, se, nboot=500, grid=None):
    sizes = sorted({r["size"] for r in rows}); S = np.array([[r["size"] == s for s in sizes] for r in rows], float)
    B = np.array([BANDS.index(C.BAND[r["size"]]) for r in rows]); w = 1 / (se ** 2 + 0.005 ** 2)
    grp = np.array([r["cand"] for r in rows])
    grid = grid or [(l, ld) for l in (1e-1, 1, 1e1, 1e2, 1e3) for ld in (1, 1e1, 1e2, 1e3, 1e4)]
    scores = {g: cv(X, B, S, y, w, grp, *g) for g in grid}; best = min(scores, key=scores.get)
    coef, _ = fit(X, B, S, y, w, *best)
    rng = np.random.default_rng(193); ug = np.unique(grp); idx = {g: np.where(grp == g)[0] for g in ug}; bs = []
    for _ in range(nboot):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        bs.append(fit(X[pick], B[pick], S[pick], y[pick], w[pick], *best)[0])
    bs = np.array(bs); sd = bs.std(0)
    nS = len(sizes); nG = len(G); out = {}
    for gi_, g in enumerate(G):
        b0 = coef[nS + gi_]; s0 = sd[nS + gi_]
        e = {"beta": float(b0), "se": float(s0)}
        for bi, b in enumerate(BANDS):
            j = nS + nG * (1 + bi) + gi_
            tot = bs[:, nS + gi_] + bs[:, j]
            e[b] = {"beta": float(b0 + coef[j]), "se": float(tot.std()), "dev": float(coef[j]), "dev_se": float(sd[j])}
        out[g] = e
    return {"lam": best, "cv": {f"{k[0]:g},{k[1]:g}": v for k, v in scores.items()}, "groups": out, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("pairs"); ap.add_argument("out")
    ap.add_argument("--pred"); ap.add_argument("--nboot", type=int, default=500)
    a = ap.parse_args()
    M = C.load_menu(a.snap); km = C.KM(M)
    rows = [json.loads(l) for l in open(a.pairs)]
    rows = [r for r in rows if r["size"] in C.BAND and r.get("se") is not None]
    G, X = design(rows, a.snap, km, M)
    res = {}
    y = np.array([r["logr"] for r in rows]); se = np.array([r["se"] for r in rows])
    res["i"] = run(rows, X, G, y, se, a.nboot)
    if a.pred:
        P = json.load(open(a.pred)); keep = [j for j, r in enumerate(rows) if r["tag"] in P]
        r2 = [rows[j] for j in keep]; yp = np.array([P[r["tag"]] for r in r2])
        res["ii"] = run(r2, X[keep], G, y[keep] - yp, se[keep], a.nboot)
        res["fit"] = run(r2, X[keep], G, yp, se[keep], a.nboot)
        res["slope"] = {}                                       # per band: weighted slope of measured on predicted log r
        for b in BANDS:
            m = np.array([C.BAND[r["size"]] == b for r in r2]); w = 1 / (se[keep][m] ** 2 + 0.005 ** 2)
            xp, ym = yp[m], y[keep][m]; xm = np.sum(w * xp) / w.sum(); ymm = np.sum(w * ym) / w.sum()
            res["slope"][b] = float(np.sum(w * (xp - xm) * (ym - ymm)) / np.sum(w * (xp - xm) ** 2))
        print("slope measured ~ fit.py predicted, per band:", res["slope"])
    json.dump(res, open(a.out, "w"), indent=1)
    for v in [k for k in res if k != "slope"]:
        print(f"== version {v}: n {res[v]['n']} lam {res[v]['lam']}")
        L = []
        for g, e in res[v]["groups"].items():
            L.append((abs(e["beta"] / e["se"]) if e["se"] else 0, g, "all", e["beta"], e["se"]))
            for b in BANDS: L.append((abs(e[b]["beta"] / e[b]["se"]) if e[b]["se"] else 0, g, b, e[b]["beta"], e[b]["se"]))
        for z, g, b, be, s in sorted(L, reverse=True)[:15]: print(f"  {g:16s} {b:4s} beta {be:+.3f} se {s:.3f} |z| {z:.1f}")


def flags(res, zmin=3.0):
    """RULES-193 §3 flags: version ii |z| >= zmin (direction = -sign beta_ii), or version i |z| >= zmin with sign opposite
    fit.py's coefficient (direction = -sign beta_i); per group x band (bands only, the pooled 'all' is not a test size)"""
    out = []
    for g, e in res["ii"]["groups"].items():
        for b in BANDS:
            x = e[b]
            if x["se"] and abs(x["beta"] / x["se"]) >= zmin:
                out.append({"g": g, "band": b, "ver": "ii", "z": abs(x["beta"] / x["se"]), "beta": x["beta"], "dir": -1 if x["beta"] > 0 else 1})
    for g, e in res["i"]["groups"].items():
        for b in BANDS:
            x = e[b]; f = res["fit"]["groups"][g][b]
            if x["se"] and abs(x["beta"] / x["se"]) >= zmin and np.sign(x["beta"]) != np.sign(f["beta"]):
                out.append({"g": g, "band": b, "ver": "i", "z": abs(x["beta"] / x["se"]), "beta": x["beta"], "fit_beta": f["beta"],
                            "dir": -1 if x["beta"] > 0 else 1})
    return sorted(out, key=lambda f: -f["z"])


if __name__ == "__main__":
    main()
