# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pool187.py -- partial pooling over the search's bands (RULES-187 arm c). A subclass of search/v3/fit.py's SModel:
  shared   the parameters fit.py fits (theta, c0, embedding, q, kappa, family factors, slice offsets, xscale)
  per band the deviations dtheta (per group x depth knot, head included), dq (log q), dkappa (log kappa) of the
           band's rows, with a Gaussian shrinkage prior N(0, sd_dev^2) on each (sd_dev = 0.1 / 0.3 / 1.0 across arms)
A row's band is its size's band (band_of_size: the driver's fit_groups b1 1.25-2.5, b2 3-4, b3 4.5-6, b4 7; the cut
points put every size of a prior row in one band). With every deviation at 0 the model is fit.py's (tested in test_187.py).

  arm c = PoolModel(build_pool(M, obs, opts, sd_dev)).fit(); band predictions via tables_band(x, k)
"""
import math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import fit as FT             # noqa: E402
import kfit3 as K3           # noqa: E402

NB = 4                       # b1..b4 (the driver's fit_groups)


def band_of_size(s):
    """driver fit_groups: b1 = 1.25-2.5, b2 = 3-4, b3 = 4.5-6, b4 = 7 (cut points between them)"""
    s = float(s or 0)
    return 0 if s < 2.75 else 1 if s < 4.25 else 2 if s < 6.5 else 3


class PoolData(FT.SData):
    def row(self, o):
        r = super().row(o); r["band"] = band_of_size(o.get("size")); return r


class PoolModel(FT.SModel):
    def __init__(self, D, sd_dev, **kw):
        super().__init__(D, **kw)
        self.sd_dev = float(sd_dev)
        self.band = np.array([r["band"] for r in D.rows], int)

    def layout(self):
        L = super().layout()                               # shared parameters (fit.py's), self.N includes lamx
        s0 = self.N; nk = len(self.KK)
        L["dth"] = slice(s0, s0 + NB * self.NTH)
        L["dq"] = slice(L["dth"].stop, L["dth"].stop + NB)
        L["dka"] = slice(L["dq"].stop, L["dq"].stop + NB * nk)
        self.N = L["dka"].stop
        return L

    def band_unpack(self, x, k):
        """the shared parameters with band k's deviations applied"""
        L = self.layout(); u = dict(self.unpack(x))
        u["theta"] = u["theta"] + x[L["dth"]].reshape(NB, self.NTH)[k]
        u["q"] = u["q"] * math.exp(x[L["dq"]][k])
        u["kap"] = u["kap"] * np.exp(x[L["dka"]].reshape(NB, len(self.KK))[k])
        return u

    def _P(self, x, u, ii):
        """kfit3.Model.predict_rows for rows ii with one band's parameters u (same arithmetic)"""
        D = self.D; self._mats()
        s = np.exp(self.A @ u["theta"] + u["beta"] * self.p)
        nv2 = len(self.v2fams); PH = np.zeros((len(self.FL), len(K3.GL)))
        for j, f in enumerate(self.v2fams): PH[1 + j, :] = u["phi2"][f]
        for j, f in enumerate(self.v3fams): PH[1 + nv2 + j, :] = u["phi3"][f] + u["dev3"][j]
        FI = self.FI[ii]; ph = PH[FI, self.G[None, :]]
        rel = np.where(D.eb > 0, self.EV[ii] * np.exp(ph) / np.where(D.eb > 0, D.eb, 1), 0)
        c = s * D.eb * rel ** u["q"]
        SW = self.SW[ii]
        if self.swk:
            f = np.ones_like(SW)
            for k, v in u["swj"].items(): f[SW == k] = v
            c = c * f
        X = c @ self.LM; kl = self.KA @ u["kap"]
        P = u["c0"] + u["emb"][self.LVI[ii]]
        for l in range(X.shape[1]): P = P * (1 + kl[l] * X[:, l]) + X[:, l]
        P = np.maximum(P, 1e-9)
        if u["sat"]: P = u["sat"] * -np.expm1(-P / u["sat"])
        return P * np.exp(u["soff"][self.SLI[ii]])

    def predict_rows(self, x, idx):
        idx = np.asarray(idx, int); out = np.empty(len(idx)); bd = self.band[idx]
        for k in np.unique(bd):
            m = bd == k; ii = idx[m]; u = self.band_unpack(x, int(k))
            P = self._P(x, u, ii)
            if self.xscale:                                 # fit.py's x-part rows: exp(soff) P^lambda
                so = np.exp(u["soff"][self.SLI[ii]]); P = np.where(self.XM[ii], so * (P / so) ** self.lam(x), P)
            out[m] = P
        return out

    def residuals(self, x, idx):
        L = self.layout()
        return np.concatenate([super().residuals(x, idx), x[L["dth"]] / self.sd_dev, x[L["dq"]] / self.sd_dev,
                               x[L["dka"]] / self.sd_dev])

    def tables_band(self, x, k):
        """fit.SModel.tables for band k: the band's s, q and kappa; everything else is shared"""
        u = self.band_unpack(x, k); tb = super().tables(x)
        tb["s"] = np.exp(self.A @ u["theta"] + u["beta"] * self.p); tb["q"] = u["q"]; tb["kl"] = self.KA @ u["kap"]
        return tb

    def predict_pair(self, x, c, a, k):
        """log(KL(c) / KL(a)) at band k (no xscale: the driver's pred_logr scale)"""
        tb = self.tables_band(x, k)
        return math.log(self.predict(x, c, tb) / self.predict(x, a, tb))


def build_pool(M, obs, opts, sd_dev):
    opts = opts or {}
    D = PoolData(M, obs)
    return PoolModel(D, sd_dev, tau=opts.get("tau", 0.02), phi_sd=opts.get("phi_sd", 1.0), beta=str(opts.get("beta", "0")),
                     theta_sd=opts.get("theta_sd", 4.0), v3_group_sd=opts.get("v3_group_sd", 0.3), smooth=opts.get("smooth", 0.5),
                     xscale=bool(opts.get("xscale", False)))
