# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fit202.py -- fit.py + two optional one-parameter terms (RULES-floor202, #202):

  opts {"floor": true}   fitfloor: KLD = exp(soff) * (F + P), prior log F ~ N(log floor_mu, floor_sd) (4e-4, 1.0). F is the
                         error no allocation removes (A8 activations); it sits outside the depth compounding, unlike c0.
  opts {"sat": true}     fitsat (amendment 1): KLD = exp(soff) * P / (1 + P / S), prior log S ~ N(log sat_mu, sat_sd) (1.0, 1.5):
                         linear in P while P << S, bounded by S. (Not kfit3's own "sat" form, which stays off.)
  (both: exp(soff) * (F + P / (1 + P / S)); x-part rows with opts xscale: the bracket to the power lambda, as fit.py's P^lambda)

Everything else is fit.py's SModel (labels, anchors, weights, priors). Predictions for moves (moves.Pred -> kl_from_X /
grad_X) are the bracket on the calib scale.

Neither opt: fit.py's own code path, untouched (same output as fit.py; checked on a live b3 input).

With a term on, two things are faster than fit.py (same model, checked against kfit3's predict_rows in `selftest`):
the Jacobian is fit.py's 2-point finite difference for every parameter except the slice offsets, whose columns are exact
(the residuals are linear in them; 368 of 474 parameters on the live b3 inputs), and the per-row power table
(err phi / eb)^q is cached between evaluations that do not move q or the family factors. opts "fastjac": false gives
scipy's own 2-point Jacobian.

  python3 fit202.py IN.json OUT.json [key=json ...] [x0=FIT.out.json]   (IN as fit.py's; key=json overrides opts, e.g.
                                         floor=true; x0: a fit.py / fit202 vector of the same rows, the new terms appended)
  python3 fit202.py selftest IN.json     fast P vs kfit3's on random parameter vectors; Jacobian vs scipy's 2-point"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "refit"))
import fit as FT              # noqa: E402
import menu as MN             # noqa: E402
from fit import least_squares  # noqa: E402

FLOOR_MU, FLOOR_SD = 4e-4, 1.0
SAT_MU, SAT_SD = 1.0, 1.5


class FModel(FT.SModel):
    def __init__(self, D, floor=False, sat202=False, floor_mu=FLOOR_MU, floor_sd=FLOOR_SD, sat_mu=SAT_MU, sat_sd=SAT_SD, fastjac=True, **kw):
        super().__init__(D, **kw)
        assert not self.sat and not self.swk                            # kfit3's own saturation / swj factors: not in the search model
        self.floor, self.sat202 = bool(floor), bool(sat202)
        self.floor_mu, self.floor_sd, self.sat_mu, self.sat_sd, self.fastjac = float(floor_mu), float(floor_sd), float(sat_mu), float(sat_sd), bool(fastjac)
        self._pc = {}

    def layout(self):
        L = super().layout()
        if getattr(self, "floor", False): L["lF"] = slice(self.N, self.N + 1); self.N += 1
        if getattr(self, "sat202", False): L["lS"] = slice(self.N, self.N + 1); self.N += 1
        return L

    def F(self, x): return math.exp(x[self.layout()["lF"]][0]) if self.floor else 0.0

    def S(self, x): return math.exp(x[self.layout()["lS"]][0]) if self.sat202 else None

    def bracket(self, P, F, S):
        """the allocation-dependent part on the calib scale: F + P / (1 + P / S)"""
        if S: P = P / (1 + P / S)
        return P + F

    def rawP(self, x, idx):
        """kfit3.Model.predict_rows without the slice offset (same arithmetic; the power table is cached)"""
        self._mats(); u = self.unpack(x); D = self.D; idx = np.asarray(idx, int)
        PH = np.zeros((len(self.FL), len(FT.GL)))
        for j, f in enumerate(self.v3fams): PH[1 + len(self.v2fams) + j, :] = u["phi3"][f] + u["dev3"][j]
        key = (float(u["q"]), PH.tobytes(), len(idx), int(idx[0]) if len(idx) else -1, int(idx[-1]) if len(idx) else -1)
        E = self._pc.get(key)
        if E is None:
            ph = PH[self.FI[idx], self.G[None, :]]
            rel = np.where(D.eb > 0, self.EV[idx] * np.exp(ph) / np.where(D.eb > 0, D.eb, 1), 0)
            E = D.eb * rel ** u["q"]
            if len(self._pc) >= 2: self._pc.pop(next(iter(self._pc)))
            self._pc[key] = E
        s = np.exp(self.A @ u["theta"] + u["beta"] * self.p)
        X = (E * s) @ self.LM; kl = self.KA @ u["kap"]
        P = u["c0"] + u["emb"][self.LVI[idx]]
        for l in range(X.shape[1]): P = P * (1 + kl[l] * X[:, l]) + X[:, l]
        return np.maximum(P, 1e-9), np.exp(u["soff"][self.SLI[idx]])

    def predict_rows(self, x, idx):
        idx = np.asarray(idx, int)
        P, so = self.rawP(x, idx); B = self.bracket(P, self.F(x), self.S(x))
        if not self.xscale: return so * B
        return np.where(self.XM[idx], so * B ** self.lam(x), so * B)

    def x0(self, head=True):
        x = super().x0(head); L = self.layout()
        if self.floor: x[L["lF"]] = math.log(self.floor_mu)
        if self.sat202: x[L["lS"]] = math.log(self.sat_mu)
        return x

    def residuals(self, x, idx):
        r = [super().residuals(x, idx)]; L = self.layout()
        if self.floor: r.append([(x[L["lF"]][0] - math.log(self.floor_mu)) / self.floor_sd])
        if self.sat202: r.append([(x[L["lS"]][0] - math.log(self.sat_mu)) / self.sat_sd])
        return np.concatenate(r)

    def jac(self, x, idx):
        """2-point finite differences (scipy's step) for all but the slice offsets; their columns are exact"""
        L = self.layout(); r0 = self.residuals(x, idx); J = np.zeros((len(r0), self.N))
        so = L["soff"]; h0 = np.finfo(float).eps ** 0.5
        for j in range(self.N):
            if so.start <= j < so.stop: continue
            h = h0 * (1.0 if x[j] >= 0 else -1.0) * max(1.0, abs(x[j])); x1 = x.copy(); x1[j] += h
            J[:, j] = (self.residuals(x1, idx) - r0) / (x1[j] - x[j])
        if so.stop > so.start:
            x1 = x.copy(); x1[so] += 1.0; d = self.residuals(x1, idx) - r0   # linear in soff: exact
            n = len(idx); sl = self.SLI[np.asarray(idx, int)]
            for i in np.nonzero(sl > 0)[0]: J[i, so.start + sl[i] - 1] = d[i]
            pr = n + np.nonzero(np.abs(d[n:]) > 1e-12)[0]              # the offsets' prior rows, in parameter order
            assert len(pr) == so.stop - so.start, (len(pr), so)
            for k, i in enumerate(pr): J[i, so.start + k] = d[i]
        return J

    def extend(self, xi):
        """a start vector from a model without (some of) the new terms: they are appended at their prior means"""
        if xi is None: return None
        xi = list(xi); base = self.N - int(self.floor) - int(self.sat202)
        if len(xi) == base: xi = xi + ([math.log(self.floor_mu)] if self.floor else []) + ([math.log(self.sat_mu)] if self.sat202 else [])
        return xi

    def fit(self, idx=None, xi=None):
        idx = list(range(len(self.D.rows))) if idx is None else idx
        self.layout()
        kw = {"jac": (lambda x: self.jac(x, idx))} if self.fastjac else {}
        f = lambda x0: least_squares(lambda x: self.residuals(x, idx), x0, method="trf", x_scale="jac",
                                     max_nfev=int(os.environ.get("FIT_NFEV", 300)), **kw)
        xi = self.extend(xi)
        if xi is not None and len(xi) == self.N: r = f(np.asarray(xi, float))
        else:
            r = f(self.x0())
            if self.head_init():
                r2 = f(self.x0(head=False))
                if r2.cost < r.cost: r = r2
        self.res = r
        return r.x

    def par_se(self, name):
        """posterior sd of log F / log S (Laplace on the fit's Jacobian, as posterior(): flat directions carry no variance)"""
        J = np.asarray(self.res.jac); H = J.T @ J; w, V = np.linalg.eigh(H)
        wi = np.where(w > 1e-8 * w.max(), 1.0 / np.maximum(w, 1e-300), 0.0)
        k = self.layout()[name].start
        return float(np.sqrt(np.sum(V[k] ** 2 * wi)))

    def tables(self, x):
        tb = super().tables(x); tb["F"] = self.F(x); tb["S"] = self.S(x)
        return tb

    def kl_from_X(self, tb, X, emb_label):
        return self.bracket(super().kl_from_X(tb, X, emb_label), tb.get("F", 0.0), tb.get("S"))

    def grad_X(self, tb, X, emb_label):
        """(dP / dX_l, D) with d log KL = dP / D (moves.Pred.lin divides the gradient by the second value)"""
        g, P = super().grad_X(tb, X, emb_label); S = tb.get("S"); F = tb.get("F", 0.0)
        if not S: return g, P + F
        return g, self.bracket(P, F, S) * (S + P) ** 2 / S ** 2

    def dump(self, x):
        d = super().dump(x); d["model"] = "kfit3/search-v3+202" + ("+floor" if self.floor else "") + ("+sat" if self.sat202 else "")
        d["F"] = self.F(x); d["S"] = self.S(x)
        return d


def on(opts): return bool((opts or {}).get("floor") or (opts or {}).get("sat"))


def build(M, obs, opts=None):
    opts = opts or {}
    if not on(opts): return FT.build(M, obs, opts)
    D = FT.SData(M, obs)
    return FModel(D, floor=opts.get("floor", False), sat202=opts.get("sat", False), floor_mu=opts.get("floor_mu", FLOOR_MU),
                  floor_sd=opts.get("floor_sd", FLOOR_SD), sat_mu=opts.get("sat_mu", SAT_MU), sat_sd=opts.get("sat_sd", SAT_SD), fastjac=opts.get("fastjac", True),
                  tau=opts.get("tau", 0.02), phi_sd=opts.get("phi_sd", 1.0), beta=str(opts.get("beta", "0")),
                  theta_sd=opts.get("theta_sd", 4.0), v3_group_sd=opts.get("v3_group_sd", 0.3), smooth=opts.get("smooth", 0.5),
                  xscale=bool(opts.get("xscale", False)))


def fit_obs(M, obs, opts=None, x0=None, ndraw=0, seed=0):
    """-> (model, x, draws, report); no term on: fit.fit_obs itself"""
    if not on(opts): return FT.fit_obs(M, obs, opts, x0, ndraw, seed)
    Mo = build(M, obs, opts); x = Mo.fit(xi=x0)
    pr = Mo.predict_rows(x, list(range(len(Mo.D.rows)))); km = np.array([r["kld"] for r in Mo.D.rows])
    lr = np.log(pr / km); se = np.array([r["se"] for r in Mo.D.rows])
    rep = {"rows": len(obs), "labels": Mo.NS, "params": Mo.N, "rmse_log": float(np.sqrt(np.mean(lr ** 2))) if len(lr) else None,
           "chi2_per_row": float(np.mean(lr ** 2 / (se ** 2 + Mo.tau ** 2))) if len(lr) else None, "cost": float(Mo.res.cost), "nfev": int(Mo.res.nfev),
           "F": Mo.F(x) if Mo.floor else None, "logF_se": Mo.par_se("lF") if Mo.floor else None,
           "S": Mo.S(x), "logS_se": Mo.par_se("lS") if Mo.sat202 else None}
    return Mo, x, Mo.posterior(x, ndraw, seed), rep


def selftest(inp):
    from scipy.optimize._numdiff import approx_derivative
    d = json.load(open(inp)); M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
    opts = d.get("opts") or {}; Ma = FT.build(M, d["obs"], opts); Ma.layout(); idx = list(range(len(Ma.D.rows)))
    rng = np.random.default_rng(0); xa = Ma.x0() + 0.05 * rng.standard_normal(Ma.N); ok = True
    xa[Ma.layout()["emb"]] = np.abs(xa[Ma.layout()["emb"]]) * 1e-3     # keep P off its 1e-9 clamp (grad_X ignores the clamp, as fit.py's)
    for o in ({"floor": True}, {"sat": True}, {"floor": True, "sat": True}):
        Mf = build(M, d["obs"], dict(opts, **o)); Mf.layout(); x = np.array(Mf.extend(xa)); L = Mf.layout()
        if Mf.floor: x[L["lF"]] = -40.0                               # F -> 0, S -> inf: must equal fit.py's rows
        if Mf.sat202: x[L["lS"]] = 40.0
        pa, pf = Ma.predict_rows(xa, idx), Mf.predict_rows(x, idx); e = float(np.max(np.abs(np.log(pf / pa)))); ok &= e < 1e-7
        x = np.array(Mf.extend(xa)); J = Mf.jac(x, idx); J2 = approx_derivative(lambda v: Mf.residuals(v, idx), x, method="2-point")
        ej = float(np.max(np.abs(J - J2)) / max(np.max(np.abs(J2)), 1e-30)); ok &= ej < 1e-5
        # moves: kl_from_X / grad_X against a numeric derivative of the row prediction's bracket
        tb = Mf.tables(x); a = d["obs"][0]["assign"]; X = Mf.layer_X(tb, a); el = Mf.D.emb_label(a)
        g, Dn = Mf.grad_X(tb, X, el); k0 = Mf.kl_from_X(tb, X, el); eg = 0.0
        for l in (0, len(X) // 2, len(X) - 1):
            X1 = X.copy(); h = 1e-5 * float(X.max()); X1[l] += h
            num = (math.log(Mf.kl_from_X(tb, X1, el)) - math.log(k0)) / h; eg = max(eg, abs(num - g[l] / Dn) / max(abs(num), 1e-300))
        ok &= eg < 1e-3
        print(o, "max |log pred fast / fit.py| %.2e" % e, "| jac vs scipy 2-point (rel) %.2e" % ej, "| grad_X vs numeric (rel) %.2e" % eg)
    print("selftest", "OK" if ok else "FAIL"); return 0 if ok else 1


def main(argv):
    if argv[0] == "selftest": return selftest(argv[1])
    d = json.load(open(argv[0]))
    if len(argv) > 2:                                                 # fit202.py IN OUT key=json ...: opts overrides
        d["opts"] = dict(d.get("opts") or {}, **{k: json.loads(v) for k, v in (a.split("=", 1) for a in argv[2:] if not a.startswith("x0="))})
        for a in argv[2:]:
            if a.startswith("x0="): d["x0"] = json.load(open(a[3:]))["x"]
    M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
    Mo, x, draws, rep = fit_obs(M, d["obs"], d.get("opts"), d.get("x0"), int(d.get("ndraw", 0)), int(d.get("seed", 0)))
    out = {"report": rep, "model": Mo.dump(x), "x": x.tolist(), "draws": [v.tolist() for v in draws], "obs_tags": [o["tag"] for o in d["obs"]],
           "opts": d.get("opts")}
    json.dump(out, open(argv[1] + ".part", "w")); os.replace(argv[1] + ".part", argv[1])
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
