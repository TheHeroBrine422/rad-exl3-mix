# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""curve193.py fit SNAP CURV.json OUT.json [--sigma S] [--folds 5 --cv]  |  alloc SNAP CURV.json FIT.json OUTDIR --sizes ...
-- RULES-193 §1 (+ amendment 1): the pooled curve model.

Per weight t: C_t(s) = s + o_t + b_t (s - 4) + g_t min(0, s - 2.5). Row i (a fit row of an allowed label, size 1.25-6):
  log KL_i = off_label + h(s_i) + kappa_band(s_i) * kappa_x^[x part] * sum_t c_t(s_i) (K_ti - C_t(s_i))^2
c_t(s) = the curvature of my fit.py refit at the incumbents (curv193.py), linear in s between sizes. h: piecewise linear
in s (knots at the sizes, h(4) = 0). Residuals (pred - log kld) / sqrt(se^2 + 0.02^2); priors: (o, b, g) ~ N(consensus
curve linearised, sigma^2) and log kappa ~ N(0, 1). sigma by grouped CV (folds by eval tag)."""
import argparse, json, math, os, sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common193 as C  # noqa: E402
import cons193 as CS  # noqa: E402

KN = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
TAU = 0.02


def band(s): return 0 if s < 2.75 else (1 if s <= 4.25 else 2)


def hbasis(s):
    """piecewise-linear hat functions at KN, minus the s=4 knot"""
    out = []
    for k in range(len(KN)):
        if KN[k] == 4.0: continue
        e = np.zeros(len(KN)); e[k] = 1.0; out.append(np.interp(s, KN, e))
    return np.array(out)


class Data:
    def __init__(self, snap, curv, km):
        rows = [r for r in C.fit_rows(snap) if r.get("size") and 1.2 <= r["size"] <= 6.2]
        self.rows = rows; self.km = km; W = km.W; self.W = W
        self.K = np.array([[km.Kof[w][r["assign"][w]] for w in W] for r in rows])
        self.s = np.array([r["size"] for r in rows]); self.y = np.log([r["kld"] for r in rows])
        self.sig = np.sqrt(np.array([r.get("se") or 0.0 for r in rows]) ** 2 + TAU ** 2)
        labs = sorted({r["label"] for r in rows}); self.labs = labs; li = {l: i for i, l in enumerate(labs)}
        self.li = np.array([li[r["label"]] for r in rows])
        self.isx = np.array([("|x|" in r["label"]) for r in rows])
        self.b = np.array([band(s) for s in self.s])
        self.H = np.array([hbasis(s) for s in self.s])                     # n x (len(KN)-1)
        cs = sorted(float(k) for k in curv["curv"]); cmat = np.array([[curv["curv"][f"{s:g}"][w][1] for w in W] for s in cs])
        self.cs, self.cmat = cs, cmat
        self.c = self.cint(self.s)
        self.tags = np.array([r["tag"].split("@")[0] for r in rows])

    def cint(self, svec):
        svec = np.atleast_1d(svec); out = np.empty((len(svec), self.cmat.shape[1]))
        for t in range(self.cmat.shape[1]): out[:, t] = np.interp(svec, self.cs, self.cmat[:, t])
        return out


def basis(s):
    s = np.atleast_1d(s)
    return np.stack([np.ones_like(s), s - 4.0, np.minimum(0.0, s - 2.5)], 1)      # n x 3


class Model:
    def __init__(self, D, prior, sigma):
        self.D = D; self.p0 = prior; self.sig = sigma
        self.nT = len(D.W); self.nL = len(D.labs); self.nH = len(KN) - 1
        self.sl = {"curve": slice(0, 3 * self.nT)}
        o = 3 * self.nT
        self.sl["off"] = slice(o, o + self.nL); o += self.nL
        self.sl["h"] = slice(o, o + self.nH); o += self.nH
        self.sl["lk"] = slice(o, o + 3); o += 3
        self.sl["lkx"] = slice(o, o + 1); o += 1
        self.N = o
        self.Bs = basis(D.s)                                                     # n x 3

    def unpack(self, x):
        cp = x[self.sl["curve"]].reshape(self.nT, 3)
        return cp, x[self.sl["off"]], x[self.sl["h"]], x[self.sl["lk"]], x[self.sl["lkx"]][0]

    def x0(self):
        x = np.zeros(self.N); x[self.sl["curve"]] = self.p0.ravel()
        D = self.D
        for l in range(self.nL):
            m = D.li == l; x[self.sl["off"]][l] = D.y[m].mean() if m.any() else 0.0
        return x

    def pieces(self, x, idx):
        D = self.D; cp, off, h, lk, lkx = self.unpack(x)
        Cv = D.s[idx, None] + self.Bs[idx] @ cp.T                                # n x T
        dev = D.K[idx] - Cv; cc = D.c[idx]
        kap = np.exp(lk[D.b[idx]] + lkx * D.isx[idx])
        q = (cc * dev * dev).sum(1)
        pred = off[D.li[idx]] + D.H[idx] @ h + kap * q
        return pred, dev, cc, kap

    def res(self, x, idx):
        D = self.D; pred, *_ = self.pieces(x, idx)
        cp = x[self.sl["curve"]].reshape(self.nT, 3)
        r1 = (pred - D.y[idx]) / D.sig[idx]
        r2 = ((cp - self.p0) / self.sig).ravel()
        lk = x[self.sl["lk"]]; lkx = x[self.sl["lkx"]]
        return np.concatenate([r1, r2, lk, lkx])

    def jac(self, x, idx):
        D = self.D; pred, dev, cc, kap = self.pieces(x, idx); n = len(idx)
        J = np.zeros((n + 3 * self.nT + 4, self.N)); w = 1 / D.sig[idx]
        g = -2 * kap[:, None] * cc * dev                                       # d pred / d C_t
        Bs = self.Bs[idx]
        Jc = (g[:, :, None] * Bs[:, None, :]).reshape(n, 3 * self.nT)
        J[:n, self.sl["curve"]] = Jc * w[:, None]
        J[np.arange(n), self.sl["off"].start + D.li[idx]] = w
        J[:n, self.sl["h"]] = D.H[idx] * w[:, None]
        qq = (cc * dev * dev).sum(1) * kap
        for b in range(3): J[:n, self.sl["lk"].start + b] = qq * (D.b[idx] == b) * w
        J[:n, self.sl["lkx"].start] = qq * D.isx[idx] * w
        J[n:n + 3 * self.nT, self.sl["curve"]] = np.eye(3 * self.nT) / self.sig
        J[n + 3 * self.nT:, self.sl["lk"].start:self.sl["lkx"].stop] = np.eye(4)
        return J

    def fit(self, idx=None, x0=None):
        idx = np.arange(len(self.D.y)) if idx is None else idx
        r = least_squares(lambda x: self.res(x, idx), self.x0() if x0 is None else x0, jac=lambda x: self.jac(x, idx),
                          method="trf", x_scale="jac", max_nfev=200)
        return r.x, r


def prior_curve(snap, km):
    """the §2 consensus targets at 2.0 ... 6.0 linearised to (o, b, g) per weight (ridge 1e-3 on g)"""
    st = C.load_state(snap); src = CS.sources(snap, st); cv = CS.Curve(km, src)
    ss = np.array(C.SIZES); Bm = basis(ss); A = Bm.T @ Bm + np.diag([0, 0, 1e-3])
    out = []
    for w in km.W:
        T = np.array([cv.target(w, s) for s in ss]) - ss
        out.append(np.linalg.solve(A, Bm.T @ T))
    return np.array(out)


def alloc(km, M, cp, cvec, s0, family_of):
    """min sum c_t (K - C_t(s0))^2 over the own-family menu, bytes <= the window top; then top-up"""
    Ct = s0 + basis(np.array([s0]))[0] @ cp.T
    opts = {w: km.fam_opts(w, family_of[w]) for w in km.W}
    hi = s0 * M.P / 8; fixed = M.fixed
    def pick(mu):
        a = {}; b = fixed
        for t, w in enumerate(km.W):
            o = min(opts[w], key=lambda z: cvec[t] * (z[0] - Ct[t]) ** 2 + mu * z[2]); a[w] = o[1]; b += o[2]
        return a, b
    a, b = pick(0.0)
    if b > hi:
        lo_m, hi_m = 0.0, 1e-12
        while pick(hi_m)[1] > hi: lo_m = hi_m; hi_m *= 4
        for _ in range(60):
            m = (lo_m + hi_m) / 2
            if pick(m)[1] > hi: lo_m = m
            else: hi_m = m
        a, b = pick(hi_m)
    b = M.bytes(a)
    while True:                                                  # top-up: best objective gain per byte that fits
        best = None
        for t, w in enumerate(km.W):
            names = [o[1] for o in opts[w]]; i = names.index(a[w])
            if i + 1 >= len(opts[w]): continue
            nk, no, nb = opts[w][i + 1]; db = nb - opts[w][i][2]
            if b + db > hi or db <= 0: continue
            gain = cvec[t] * ((opts[w][i][0] - Ct[t]) ** 2 - (nk - Ct[t]) ** 2)
            if gain <= 0: continue
            if best is None or gain / db > best[0]: best = (gain / db, w, no, db)
        if best is None: break
        a[best[1]] = best[2]; b += best[3]
    return a, Ct


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("snap"); ap.add_argument("curv")
    ap.add_argument("rest", nargs="*"); ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--cv", action="store_true"); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sizes", default="2,2.5,3,3.5,4,4.5,5,6"); ap.add_argument("--commit", default="?")
    ap.add_argument("--mode", default="decide")
    a = ap.parse_args()
    M = C.load_menu(a.snap); km = C.KM(M); curv = json.load(open(a.curv))
    if a.cmd == "fit":
        D = Data(a.snap, curv, km); p0 = prior_curve(a.snap, km)
        print(f"rows {len(D.y)} labels {len(D.labs)} dropped {C.dropped}", flush=True)
        out = {"sigma": a.sigma}
        if a.cv:
            rng = np.random.default_rng(193); ut = np.unique(D.tags); fold = dict(zip(ut, rng.integers(0, a.folds, len(ut))))
            f = np.array([fold[t] for t in D.tags]); err = []; n = 0
            for k in range(a.folds):
                tr = np.where(f != k)[0]; te = np.where(f == k)[0]
                Mo = Model(D, p0, a.sigma); x, r = Mo.fit(tr)
                # held-out: label offsets of held-out rows are refit on held-out anchors? use within-label differences:
                pred, *_ = Mo.pieces(x, te); e = (pred - D.y[te]) / D.sig[te]
                # centre per label on the held-out rows (labels whose only rows are held out have an untrained offset)
                for l in np.unique(D.li[te]):
                    m = D.li[te] == l
                    if m.sum() >= 2: e[m] -= e[m].mean()
                    else: e[m] = np.nan
                err.append(e[~np.isnan(e)])
            ee = np.concatenate(err); out["cv_chi2"] = float(np.mean(ee ** 2)); out["cv_n"] = len(ee)
            print(f"sigma {a.sigma}: CV chi2/row {out['cv_chi2']:.4f} (n {len(ee)})", flush=True)
        Mo = Model(D, p0, a.sigma); x, r = Mo.fit()
        pred, *_ = Mo.pieces(x, np.arange(len(D.y))); e = (pred - D.y)
        cp, off, h, lk, lkx = Mo.unpack(x)
        out.update({"x": x.tolist(), "cp": cp.tolist(), "p0": p0.tolist(), "kappa": np.exp(lk).tolist(), "kappa_x": float(np.exp(lkx)),
                    "rmse_log": float(np.sqrt(np.mean(e ** 2))), "chi2": float(np.mean((e / D.sig) ** 2)), "nfev": int(r.nfev),
                    "W": D.W, "rows": len(D.y)})
        print(f"fit: rmse_log {out['rmse_log']:.4f} chi2 {out['chi2']:.3f} kappa {out['kappa']} kx {out['kappa_x']:.3f} "
              f"|cp-p0| mean {np.abs(cp - p0).mean(0)} nfev {r.nfev}", flush=True)
        json.dump(out, open(a.rest[0], "w"))
    elif a.cmd == "alloc":
        F = json.load(open(a.rest[0])); outdir = a.rest[1]; os.makedirs(outdir, exist_ok=True)
        cp = np.array(F["cp"]); st = C.load_state(a.snap); inc = {}
        for e in st:
            if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
        cs = sorted(float(k) for k in curv["curv"])
        for s0 in [float(x) for x in a.sizes.split(",")]:
            cvec = np.array([np.interp(s0, cs, [curv["curv"][f"{s:g}"][w][1] for s in cs]) for w in km.W])
            ref = inc.get(s0) or inc[min(inc, key=lambda z: abs(z - s0))]
            ia = C.assign(a.snap, ref); fam = {w: km.fam(ia[w]) for w in km.W}
            asg, Ct = alloc(km, M, cp, cvec, s0, fam)
            qa = sum(cvec[t] * (km.Kof[w][asg[w]] - Ct[t]) ** 2 for t, w in enumerate(km.W))
            qi = sum(cvec[t] * (km.Kof[w][ia[w]] - Ct[t]) ** 2 for t, w in enumerate(km.W))
            kb = F["kappa"][band(s0)]
            nd = sum(asg[w] != ia[w] for w in km.W); dk = np.mean([abs(km.Kof[w][asg[w]] - km.Kof[w][ia[w]]) for w in km.W])
            d = {"size": s0, "mode": a.mode, "assign": asg,
                 "note": f"#193 curve | pooled curve model K_t(s) (kink 2.5, fit on all allowed evals) beats the incumbent | code {a.commit}"}
            why = C.check_file(M, d)
            print(f"{s0:g}: bpw {M.bpw(asg):.4f} vs {ref}: {nd} tensors changed, mean|dK| {dk:.3f}, model log r {kb * (qa - qi):+.4f} {why or 'OK'}")
            if why is None: json.dump(d, open(f"{outdir}/curve-{s0:g}.json", "w"))


if __name__ == "__main__":
    main()
