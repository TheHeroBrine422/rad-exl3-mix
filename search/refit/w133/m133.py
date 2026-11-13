# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""m133.py -- #173 (w133): the candidate sensitivity models (RULES-173.md), all on fit.py's data interface.

XModel = search/v3/fit.py's SModel (kfit3 as the search fits it), subclassed, not edited. With ext = {} it is M0:
predict_rows reproduces SModel's to rounding (checked by `python3 m133.py selftest DS.json.gz`), and it is fitted by
SModel.fit (same starts, same least_squares).
ext keys (M1 / M3 terms; each adds parameters at the end of the layout and a prior residual):
  qg       per-group K-curve exponent: q_g = q exp(dq_g), dq_g ~ N(0, 0.3) (head: its own dq)
  headk    the head's own compounding factor kappa_h (log kappa_h ~ N(log kappa(NL), 3)) -- head x trunk
  sib      within-layer mixer x FFN product: X_l = A_l + F_l + rho A_l F_l, rho = exp(r), r ~ N(log 10, 3)
  adj      neighbour layers: X_l += nu X_{l-1} X_l, nu = exp(n), n ~ N(log 10, 3)
  lam2     per-part non-linear compression: x part log pred = soff + lam logP + 1/2 lam1 (logP - m)^2; c part
           soff + logP + 1/2 mu1 (logP - m)^2 (lam1, mu1 ~ N(0, 0.5); m = log median c-part KLD of the rows)
  re       per-tensor random effects in log s_w: u_w ~ N(0, re_sd[group]^2) (value = sd, one for all groups, or a
           list of 7: six groups + head)
  prof     label offsets profiled out (closed-form weighted mean of log meas - part transform per label, every
           evaluation; no offset parameters) -- with ev133's "relabel": "size" (search labels split per size) the fit
           sees only within-(label, size) ratios, i.e. candidate vs incumbent, plus fit129's cross-level rows
  mode     per act-mode offsets (A8 hook, w132): rows with mode m != f16 get P_m = P (1 + a_m) + c_m (a_m, c_m ~ N(0, 1));
           only fitted when A8 rows are in the fit
Event predictions: logpart(x, a, part) = the part's log prediction without the label offset (what cancels in a ratio)."""
import math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v3")); sys.path.insert(0, os.path.join(HERE, ".."))
import fit as FT              # noqa: E402
import menu as MN             # noqa: E402

GL = FT.GL
MIX = {"attn_in", "attn_out", "ssm_in", "ssm_out"}


class XModel(FT.SModel):
    def __init__(self, D, ext=None, modes=None, **kw):
        self.ext = dict(ext or {})
        super().__init__(D, **kw)
        self.modes = modes                                             # per row act mode (None: all f16)
        self.mlist = sorted({m for m in (modes or []) if m != "f16"}) if self.ext.get("mode") else []
        kl = [r["kld"] for r in D.rows if FT.part_of(r["slice"]) != "x"] or [r["kld"] for r in D.rows] or [0.01]
        self.m0 = math.log(float(np.median(kl)))
        self.GH = np.where(D.grp >= 0, D.grp, len(GL))                 # group index, head = 6
        self.MIXM = np.array([GL[g] in MIX if g >= 0 else False for g in D.grp])
        self.FFNM = np.array([GL[g] in ("ffn_gu", "ffn_down") if g >= 0 else False for g in D.grp])

    # ---------------------------------------------------------------- parameters
    def layout(self):
        L = super().layout(); e = self.ext
        if e.get("prof"):                                              # drop the offset parameters (profiled)
            k = L["soff"].stop - L["soff"].start
            if k:
                L = {n: (sl if sl.start < L["soff"].stop else slice(sl.start - k, sl.stop - k)) for n, sl in L.items() if n != "soff"}
                L["soff"] = slice(self.N - k, self.N - k); self.N -= k
        def add(name, k):
            L[name] = slice(self.N, self.N + k); self.N += k
        if e.get("qg"): add("dq", len(GL) + 1)
        if e.get("headk"): add("kh", 1)
        if e.get("sib"): add("rho", 1)
        if e.get("adj"): add("nu", 1)
        if e.get("lam2"): add("lam1", 1); add("mu1", 1)
        if e.get("re"): add("re", len(self.D.W))
        if e.get("mode"): add("mo", 2 * len(getattr(self, "mlist", [])))
        return L

    def x0(self, head=True):
        x = super().x0(head); L = self.layout()
        if "kh" in L: x[L["kh"]] = math.log(1.0)
        if "rho" in L: x[L["rho"]] = math.log(10.0)
        if "nu" in L: x[L["nu"]] = math.log(10.0)
        return x

    def re_sd(self):
        v = self.ext.get("re")
        return np.asarray(v, float)[self.GH] if isinstance(v, (list, tuple)) else np.full(len(self.D.W), float(v))

    # ---------------------------------------------------------------- the model
    def core(self, x, EV, FI, LVI, MO=None):
        """P (calib scale, no label offset) of rows given by error matrix EV, family index FI, emb level LVI"""
        self._mats(); u = self.unpack(x); D = self.D; L = self.layout(); e = self.ext
        ls = self.A @ u["theta"] + u["beta"] * self.p
        if "re" in L: ls = ls + x[L["re"]]
        s = np.exp(ls)
        nv2 = len(self.v2fams)
        PH = np.zeros((len(self.FL), len(GL)))
        for j, f in enumerate(self.v2fams): PH[1 + j, :] = u["phi2"][f]
        for j, f in enumerate(self.v3fams): PH[1 + nv2 + j, :] = u["phi3"][f] + u["dev3"][j]
        ph = PH[FI, self.G[None, :]]
        rel = np.where(D.eb > 0, EV * np.exp(ph) / np.where(D.eb > 0, D.eb, 1), 0)
        q = u["q"] * np.exp(x[L["dq"]][self.GH]) if "dq" in L else u["q"]
        c = s * D.eb * rel ** q
        kl = self.KA @ u["kap"]
        if "kh" in L: kl = kl.copy(); kl[D.NL] = math.exp(x[L["kh"]][0])
        if "rho" in L:
            A = (c * self.MIXM) @ self.LM; F = (c * self.FFNM) @ self.LM
            X = c @ self.LM + math.exp(x[L["rho"]][0]) * A * F
        else: X = c @ self.LM
        nu = math.exp(x[L["nu"]][0]) if "nu" in L else 0.0
        P = u["c0"] + u["emb"][LVI]
        Xp = np.zeros(X.shape[0])
        for l in range(X.shape[1]):
            Xl = X[:, l] + nu * Xp * X[:, l] if nu else X[:, l]
            P = P * (1 + kl[l] * Xl) + Xl; Xp = X[:, l]
        P = np.maximum(P, 1e-9)
        if u["sat"]: P = u["sat"] * -np.expm1(-P / u["sat"])
        if "mo" in L and MO is not None and self.mlist:
            mo = x[L["mo"]]
            for j, m in enumerate(self.mlist):
                k = MO == m; P = np.where(k, P * (1 + mo[2 * j]) + abs(mo[2 * j + 1]) * 1e-3, P)
        return P

    def gpart(self, x, logP, isx):
        """the part transform of log P (no offset)"""
        L = self.layout()
        out = np.array(logP, float)
        if self.xscale: out = np.where(isx, self.lam(x) * logP, out)
        if "lam1" in L:
            d2 = 0.5 * (logP - self.m0) ** 2
            out = out + np.where(isx, x[L["lam1"]][0], x[L["mu1"]][0]) * d2
        return out

    def predict_rows(self, x, idx):
        self._mats(); idx = np.asarray(idx, int)
        MO = np.asarray(self.modes, object)[idx] if self.modes is not None else None
        P = self.core(x, self.EV[idx], self.FI[idx], self.LVI[idx], MO)
        g = self.gpart(x, np.log(P), self.XM[idx])
        if self.ext.get("prof"):
            km = np.log([self.D.rows[i]["kld"] for i in idx]); se = np.array([self.D.rows[i]["se"] for i in idx])
            w = 1.0 / (se * se + self.tau * self.tau); sl = self.SLI[idx]
            num = np.bincount(sl, w * (km - g), minlength=self.NS); den = np.bincount(sl, w, minlength=self.NS)
            return np.exp(g + (num / np.maximum(den, 1e-300))[sl])
        so = self.unpack(x)["soff"][self.SLI[idx]]
        return np.exp(so + g)

    def residuals(self, x, idx):
        r = super().residuals(x, idx); L = self.layout(); ex = []
        if "dq" in L: ex.append(x[L["dq"]] / 0.3)
        if "kh" in L:
            kn = float((self.KA @ self.unpack(x)["kap"])[self.D.NL])
            ex.append((x[L["kh"]] - math.log(max(kn, 1e-6))) / 3.0)
        if "rho" in L: ex.append((x[L["rho"]] - math.log(10.0)) / 3.0)
        if "nu" in L: ex.append((x[L["nu"]] - math.log(10.0)) / 3.0)
        if "lam1" in L: ex.append(x[L["lam1"]] / 0.5); ex.append(x[L["mu1"]] / 0.5)
        if "re" in L: ex.append(x[L["re"]] / self.re_sd())
        if "mo" in L: ex.append(x[L["mo"]] / 1.0)
        return np.concatenate([r] + ex) if ex else r

    # ---------------------------------------------------------------- moves.Pred interface (knapsack)
    def tables(self, x):
        """SModel's per-tensor factors + the extensions that change a tensor's cost (re, qg) or the compounding (headk);
        prof / lam2 / mode change offsets or the part transform only (moves rank on P)"""
        tb = super().tables(x); L = self.layout()
        if "re" in L: tb["s"] = tb["s"] * np.exp(x[L["re"]])
        if "dq" in L: tb["qv"] = tb["q"] * np.exp(x[L["dq"]][self.GH])
        if "kh" in L: tb["kl"] = tb["kl"].copy(); tb["kl"][self.D.NL] = math.exp(x[L["kh"]][0])
        if "rho" in L or "nu" in L: raise NotImplementedError("sib / adj terms are not decomposable per tensor (not frozen)")
        return tb

    def cost(self, tb, i, err, fam):
        if "qv" not in tb: return super().cost(tb, i, err, fam)
        eb = tb["eb"][i]
        if eb <= 0: return 0.0
        g = self.D.grp[i] if self.D.grp[i] >= 0 else 0
        lph = tb["ph"][fam][g] if fam in tb["ph"] else 0.0
        return tb["s"][i] * eb * (err * math.exp(lph) / eb) ** tb["qv"][i]

    # ---------------------------------------------------------------- assignments that are not rows
    def feats(self, assigns):
        """EV, FI, LVI for a list of assignment dicts"""
        self._mats(); fi = {f: i for i, f in enumerate(self.FL)}
        EV, FI, LV = [], [], []
        for a in assigns:
            e, fam, emb = self.D.features(a)
            EV.append(e); FI.append([fi.get(f, 0) for f in fam]); LV.append(self.levels.index(emb) if emb in self.levels else self.blv)
        return np.array(EV), np.array(FI), np.array(LV)

    def logpart(self, x, assigns, part, mode="f16"):
        EV, FI, LV = self.feats(assigns)
        MO = np.array([mode] * len(assigns), object)
        P = self.core(x, EV, FI, LV, MO)
        return self.gpart(x, np.log(P), np.full(len(assigns), part == "x"))


def build(M, obs, opts=None, ext=None):
    opts = opts or {}
    D = FT.SData(M, obs)
    return XModel(D, ext=ext, modes=[o.get("mode", "f16") for o in obs], tau=opts.get("tau", 0.02),
                  phi_sd=opts.get("phi_sd", 1.0), beta=str(opts.get("beta", "0")), theta_sd=opts.get("theta_sd", 4.0),
                  v3_group_sd=opts.get("v3_group_sd", 0.3), smooth=opts.get("smooth", 0.5), xscale=bool(opts.get("xscale", False)))


def selftest(dsp):
    """XModel(ext={}) == SModel on the same rows and parameters (random x near the start)"""
    import gzip, json
    d = json.load(gzip.open(dsp, "rt")); M = MN.Menu.load(os.environ.get("MENU", d["menu"]))
    W = d["W"]; A = d["assign"]
    asg = lambda t: {w: M.T[M.WI[w]]["opts"][j]["o"] for w, j in zip(W, A[t])}
    obs = [dict(tag=o["tag"], assign=asg(o["a"]), label=o["label"], kld=o["kld"], se=o["se"]) for o in d["obs"][:300] + d["prior"]]
    opts = {"tau": 0.02, "beta": "0", "xscale": True}
    S = FT.build(M, obs, opts); X = build(M, obs, opts)
    S.layout(); X.layout(); assert S.N == X.N, (S.N, X.N)
    rng = np.random.default_rng(0); x = S.x0() + 0.05 * rng.standard_normal(S.N)
    a, b = S.predict_rows(x, list(range(len(obs)))), X.predict_rows(x, list(range(len(obs))))
    print("selftest predict_rows max |log ratio|", float(np.max(np.abs(np.log(a / b)))))
    lp = X.logpart(x, [obs[0]["assign"], obs[5]["assign"]], "c"); pp = [math.log(S.predict(x, obs[0]["assign"])), math.log(S.predict(x, obs[5]["assign"]))]
    print("selftest logpart c vs SModel.predict", np.round(lp - np.array(pp), 12).tolist())
    lx = X.logpart(x, [obs[0]["assign"]], "x")[0]
    print("selftest logpart x vs lam log P", lx - S.lam(x) * pp[0])


if __name__ == "__main__":
    if sys.argv[1] == "selftest": selftest(sys.argv[2])
