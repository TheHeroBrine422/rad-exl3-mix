# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""m158.py -- (w158, #173-03, RULES-173-03) M1m: m133's XModel (= fit.py's SModel + ext terms) with
  embf "out" | "in"   the embedding priced on the menu's err (w157's floor model) with one parameter eta = exp(h),
                      h ~ N(0, 1): dE_e = err_e - err_ref (ref = the rows' commonest embedding, t8m here);
                      "in": P_0 = c0 + eta dE_e (kfit3's embedding offset, tied to err); "out": P = P_L + eta dE_e (after
                      the compounding). Every menu option is priced (no per-option offsets, no unseen-option fallback).
  v2x F               every V2-era prior sd x F (theta smoothness / shrinkage, head vs trunk, kappa second difference,
                      family-group deviation, family factor): F = 4 is RULES-173-03's M1m-noV2 (default 1 = as fit.py)
  lin2                lam2's curvature linearised outside [lo, hi] = the training rows' log KLD range (value and slope
                      continuous): F3 of the sub-1 rule
  sat (opts)          kfit3's saturation P -> C (1 - exp(-P / C)): F2 of the sub-1 rule
The rest is m133 (prof, lam2, mode, ...). With ext {} and v2x 1 it is m133.XModel (selftest)."""
import math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "w133"), os.path.join(HERE, "..", ".."), os.path.join(HERE, "..", "..", "v3")):
    if os.path.isdir(p): sys.path.insert(0, p)
import fit as FT              # noqa: E402
import m133 as MM             # noqa: E402
import menu as MN             # noqa: E402

GL = FT.GL


class M1m(MM.XModel):
    def __init__(self, D, ext=None, modes=None, v2x=1.0, **kw):
        self.v2x = float(v2x)
        super().__init__(D, ext=ext, modes=modes, **kw)
        e = self.ext.get("embf")
        if e:
            M = D.M; eo = M.T[M.WI[MN.EMB]]["opts"]; err = {o["o"]: o["err"] for o in eo}
            ref = D.base.split("-", 1)[1]; ref = D.emb_names[int(ref[1:])]
            self.dE_name = {n: err[n] - err[ref] for n in D.emb_names}
            self.levels = [f"e{j}" for j in range(len(D.emb_names))]; self.blv = self.levels.index(D.base.split("-", 1)[1])
            self.dE = np.array([self.dE_name[D.emb_names[j]] for j in range(len(D.emb_names))])
        if self.ext.get("lin2"):
            k = np.log([r["kld"] for r in D.rows]); self.l2lo, self.l2hi = float(k.min()), float(k.max())

    def layout(self):
        L = super().layout()
        if self.ext.get("embf") and hasattr(self, "dE"):
            k = L["emb"].stop - L["emb"].start                        # drop the per-option offsets
            if k:
                L = {n: (sl if sl.start < L["emb"].stop else slice(sl.start - k, sl.stop - k)) for n, sl in L.items() if n != "emb"}
                self.N -= k
            L["emb"] = slice(L["theta"].stop + 1, L["theta"].stop + 1)
            L["eta"] = slice(self.N, self.N + 1); self.N += 1
        return L

    def unpack(self, x):
        if not (self.ext.get("embf") and hasattr(self, "dE")): return super().unpack(x)
        L = self.layout(); levels = self.levels; self.levels = [levels[self.blv]]; bl = self.blv; self.blv = 0
        try: u = super().unpack(x)
        finally: self.levels, self.blv = levels, bl
        eta = math.exp(x[L["eta"]][0]); u["eta"] = eta
        u["emb"] = eta * self.dE if self.ext["embf"] == "in" else np.zeros(len(self.dE))
        u["emb_out"] = eta * self.dE if self.ext["embf"] == "out" else np.zeros(len(self.dE))
        return u

    def x0(self, head=True):
        x = super().x0(head)
        return x

    # ---------------------------------------------------------------- the model (m133.core + the "out" embedding)
    def core(self, x, EV, FI, LVI, MO=None):
        P = super().core(x, EV, FI, LVI, None)
        u = self.unpack(x)
        if self.ext.get("embf") == "out":
            P = np.maximum(P + u["emb_out"][LVI], 1e-9)
        L = self.layout()
        if "mo" in L and MO is not None and self.mlist:                # signed c (RULES-173-03: a, c ~ N(0, 1)); m133's abs(c)
            mo = x[L["mo"]]                                             # assumed an f16 reference (A8 >= f16): with A8 as the
            for j, m in enumerate(self.mlist):                          # reference the f16 twins sit below it
                k = MO == m; P = np.where(k, np.maximum(P * (1 + mo[2 * j]) + mo[2 * j + 1] * 1e-3, 1e-9), P)
        return P

    def gpart(self, x, logP, isx):
        if not self.ext.get("lin2") or "lam1" not in self.layout(): return super().gpart(x, logP, isx)
        L = self.layout(); out = np.array(logP, float)
        if self.xscale: out = np.where(isx, self.lam(x) * logP, out)
        lo, hi, m = self.l2lo, self.l2hi, self.m0
        d = np.asarray(logP, float)
        q = np.where(d < lo, 0.5 * (lo - m) ** 2 + (lo - m) * (d - lo), np.where(d > hi, 0.5 * (hi - m) ** 2 + (hi - m) * (d - hi), 0.5 * (d - m) ** 2))
        return out + np.where(isx, x[L["lam1"]][0], x[L["mu1"]][0]) * q

    def residuals(self, x, idx):
        """fit.SModel.residuals with the V2-era prior sds x v2x, then m133's ext priors and eta's"""
        D = self.D; L = self.layout(); u = self.unpack(x); f = self.v2x
        km = np.array([D.rows[i]["kld"] for i in idx]); se = np.array([D.rows[i]["se"] for i in idx])
        res = [(np.log(self.predict_rows(x, idx)) - np.log(km)) / np.sqrt(se * se + self.tau * self.tau)]
        th = u["theta"]
        for g in range(len(GL)):
            t = th[g * self.NK:(g + 1) * self.NK]; res += [np.diff(t, 2) / (self.smooth * f), (t - t.mean()) / (self.theta_sd * f)]
        res.append(np.atleast_1d((th[:-1].mean() - th[-1]) / (6.0 * f)))
        if len(self.KK) == 3: res.append(np.diff(np.log(u["kap"]), 2) / f)
        if self.v3fams:
            res.append(u["dev3"].ravel() / (self.v3sd * f)); res.append(x[L["phi3"]] / (self.phi_sd * f))
        if self.NS > 1 and not self.ext.get("prof"): res.append(x[L["soff"]] / 3.0)
        if self.xscale: res.append(x[L["lamx"]] / 0.5)
        ex = []
        if "dq" in L: ex.append(x[L["dq"]] / 0.3)
        if "lam1" in L: ex.append(x[L["lam1"]] / 0.5); ex.append(x[L["mu1"]] / 0.5)
        if "re" in L: ex.append(x[L["re"]] / self.re_sd())
        if "mo" in L: ex.append(x[L["mo"]] / 1.0)
        if "eta" in L: ex.append(x[L["eta"]] / 1.0)
        return np.concatenate(res + ex)

    # ---------------------------------------------------------------- knapsack interface
    def tables(self, x):
        tb = super().tables(x)
        if self.ext.get("embf") and hasattr(self, "dE"):
            u = self.unpack(x)
            tb["emb"] = {f"e{j}": float(v) for j, v in enumerate(u["emb"])}
            tb["emb_out"] = {f"e{j}": float(v) for j, v in enumerate(u["emb_out"])}
        if self.ext.get("lin2"): tb["lin2"] = True
        return tb

    def kl_from_X(self, tb, X, emb_label):
        P = super().kl_from_X(tb, X, emb_label)
        return max(P + tb.get("emb_out", {}).get(emb_label, 0.0), 1e-9)

    def grad_X(self, tb, X, emb_label):
        g, P = super().grad_X(tb, X, emb_label)
        return g, P + tb.get("emb_out", {}).get(emb_label, 0.0)

    def feats(self, assigns):
        self._mats(); fi = {f: i for i, f in enumerate(self.FL)}
        EV, FI, LV = [], [], []
        for a in assigns:
            e, fam, emb = self.D.features(a)
            EV.append(e); FI.append([fi.get(f, 0) for f in fam]); LV.append(self.levels.index(emb) if emb in self.levels else self.blv)
        return np.array(EV), np.array(FI), np.array(LV)


class SData2(FT.SData):
    """fit.SData with the rows' commonest embedding option first in emb_names, so the base label is "e0" (kfit3's elevel reads one
    digit: the 37-option embedding menu's t8m is option 35)"""
    def __init__(self, M, obs):
        super().__init__(M, obs)
        if self.has_emb and obs:
            names = [o["assign"][MN.EMB] for o in obs]; ref = max(set(names), key=names.count)
            self.emb_names = [ref] + [n for n in self.emb_names if n != ref]
            for r, o in zip(self.rows, obs): r["emb"] = self.emb_label(o["assign"])
            self.base = "base-e0"


def mmode(o):
    """m133's mode term takes f16 as its reference: here A8 is the reference (every prediction is A8), so A8 rows are passed
    as "f16" and the f16 twins as "x16" (their own P (1 + a) + c)"""
    return "f16" if o.get("mode", "a8") == "a8" else "x16"


def build(M, obs, opts=None, ext=None):
    opts = opts or {}; f = float(opts.get("v2x", 1.0))
    D = SData2(M, obs)
    return M1m(D, ext=ext, modes=[mmode(o) for o in obs], v2x=f, tau=opts.get("tau", 0.02),
               phi_sd=opts.get("phi_sd", 1.0), beta=str(opts.get("beta", "0")), theta_sd=opts.get("theta_sd", 4.0),
               v3_group_sd=opts.get("v3_group_sd", 0.3), smooth=opts.get("smooth", 0.5), xscale=bool(opts.get("xscale", False)),
               sat=bool(opts.get("sat", False)))
