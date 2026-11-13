# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fit.py -- the search's model: w120's kfit3 form (search/refit/kfit3.py: cost_w = s_w eb_w (phi_fam e_w / eb_w)^q,
depth compounding P_l = P_{l-1} (1 + kappa_l X_l) + X_l, KLD = P x slice offset) fitted on the search's own rows
(TODO "Search driver: how staged evals feed the fit"):
  - every eval is a row, accepted or rejected: its KLD on the documents it scored, labelled (slice, subset) -- "S|full"
    for a whole-slice row, "S|pre<k>" for a run stopped after its first k documents (the slice's fixed order);
  - each label has its own offset and holds the incumbent's KLD on exactly those documents as its anchor row, so only
    within-label ratios inform the fit;
  - inverse-variance weights: residual = log(pred / meas) / sqrt(SE^2 + tau^2), SE = the row's paired delta-method SE
    vs its anchor (anchors: SE 0), tau = the misfit floor;
  - kfit3's priors (theta smoothness / shrinkage, kappa, per-group family deviations, slice offsets) + a weak prior on
    the family factors (log phi ~ N(0, phi_sd)), which a one-family menu cannot identify;
  - Laplace posterior (J^T J at the optimum) for Thompson draws and move uncertainty.
The embedding is a level (kfit3's emb term); the head is a tensor in layer n_layers.

  python3 fit.py IN.json OUT.json      (IN: {"menu": MENU or path, "obs": [...], "opts": {...}, "x0": [...]}) -- the
                                       driver's fit_cmd (run on colo: CPU, single-threaded)
obs: {"tag", "assign": {w: option}, "label", "kld", "se"}."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "refit"))
try:
    from scipy.optimize import least_squares
except ImportError:          # the VM's python: the driver only predicts there (fits run on colo through fit_cmd)
    import types
    def least_squares(*a, **k): raise RuntimeError("scipy is needed to fit: run fit.py through fit_cmd on colo")
    _sp = types.ModuleType("scipy"); _spo = types.ModuleType("scipy.optimize"); _spo.least_squares = least_squares; _sp.optimize = _spo
    sys.modules.setdefault("scipy", _sp); sys.modules.setdefault("scipy.optimize", _spo)
import kfit3 as K3            # noqa: E402
import menu as MN             # noqa: E402

GL = K3.GL


class SData:
    """kfit3.Data's interface (W, WI, eb, layer, grp, NL, base, rows) over a menu and the search's observations"""

    def __init__(self, M, obs):
        self.M = M
        self.W = [t["w"] for t in M.T if t["w"] != MN.EMB]; self.WI = {w: i for i, w in enumerate(self.W)}
        self.NL = M.NL
        ti = {t["w"]: t for t in M.T}
        self.eb = np.array([ti[w]["opts"][len(ti[w]["opts"]) // 2]["err"] for w in self.W])
        self.layer = np.array([ti[w]["layer"] for w in self.W])
        self.grp = np.array([GL.index(ti[w]["group"]) if ti[w]["group"] in GL else -1 for w in self.W])
        self.hbar = np.array([ti[w].get("hbar") or 0.0 for w in self.W])
        self.has_emb = MN.EMB in M.WI
        self.emb_names = [o["o"] for o in M.T[M.WI[MN.EMB]]["opts"]] if self.has_emb else ["-"]
        self.rows = [self.row(o) for o in obs]
        embs = [r["emb"] for r in self.rows] or ["e0"]
        self.base = "base-" + max(set(embs), key=embs.count)          # elevel(base) = the commonest embedding label

    def emb_label(self, a):
        return f"e{self.emb_names.index(a[MN.EMB])}" if self.has_emb and MN.EMB in a else "e0"

    def features(self, a):
        """per-tensor error vector, family names, embedding label of an assignment"""
        e = np.zeros(len(self.W)); fam = []
        for i, w in enumerate(self.W):
            o = self.M.opt(w, a[w]); e[i] = o["err"]; fam.append(o.get("fam", "v3"))
        return e, fam, self.emb_label(a)

    def row(self, o):
        e, fam, emb = self.features(o["assign"])
        return {"tag": o["tag"], "e": e, "fam": fam, "emb": emb, "slice": o["label"], "kld": float(o["kld"]), "v2": False,
                "swj": None, "bpw": None, "se": float(o.get("se") or 0.0)}


class SModel(K3.Model):
    def __init__(self, D, tau=0.02, phi_sd=1.0, beta="0", **kw):
        p = np.where(D.hbar > 0, np.log(np.where(D.hbar > 0, D.hbar, 1)), 0.0)
        for g in range(len(GL)):                                        # centred per group (kfit3.prior_vector)
            m = (D.grp == g) & (D.hbar > 0)
            if m.any(): p[D.grp == g] = np.where(D.hbar[D.grp == g] > 0, p[D.grp == g] - p[m].mean(), 0.0)
        super().__init__(D, beta=beta, prior_p=p, v2wt=1.0, **kw)
        self.tau, self.phi_sd = tau, phi_sd
        # every family a row uses (kfit3 keeps only names starting "v3" as V3 families; ours all do)

    def x0(self):
        L = self.layout(); x = np.zeros(self.N)
        k0 = float(np.median([r["kld"] for r in self.D.rows])) if self.D.rows else 0.1
        x[L["theta"]] = math.log(0.7 * k0 / max(self.D.eb.sum(), 1e-30)); x[L["c0"]] = math.log(0.1 * k0); x[L["q"]] = math.log(1.0)
        x[L["kap"]] = math.log(1.0)
        hs = float(os.environ.get("FIT_HEAD_S0", 0.5))                 # w129 (#169): the head's err is its plan's head-only KL
        M = self.D.M                                                    # (~1e-3), so at the trunk's theta its cost and gradient
        if hs > 0 and MN.HEAD in M.WI and len(M.T[M.WI[MN.HEAD]]["opts"]) > 1:   # vanish and the head stays unpriced
            x[L["theta"].stop - 1] = math.log(hs)                       # (measured s_head 0.43); priced-head menus only
        if self.beta_mode == "fit": x[L["beta"]] = 1.0
        if self.sat: x[L["sat"]] = math.log(5.0)
        return x

    def residuals(self, x, idx):
        D = self.D; L = self.layout(); u = self.unpack(x)
        km = np.array([D.rows[i]["kld"] for i in idx]); se = np.array([D.rows[i]["se"] for i in idx])
        res = [(np.log(self.predict_rows(x, idx)) - np.log(km)) / np.sqrt(se * se + self.tau * self.tau)]
        sg = self.tau                                                   # kfit3's prior terms are in units of its sigma
        th = u["theta"]
        for g in range(len(GL)):
            t = th[g * self.NK:(g + 1) * self.NK]; res += [np.diff(t, 2) / self.smooth, (t - t.mean()) / self.theta_sd]
        res.append(np.atleast_1d((th[:-1].mean() - th[-1]) / 6.0))
        if len(self.KK) == 3: res.append(np.diff(np.log(u["kap"]), 2))
        if self.v3fams:
            res.append(u["dev3"].ravel() / self.v3sd); res.append(x[L["phi3"]] / self.phi_sd)
        if self.NS > 1: res.append(x[L["soff"]] / 3.0)
        del sg
        return np.concatenate(res)

    def fit(self, idx=None, xi=None):
        idx = list(range(len(self.D.rows))) if idx is None else idx
        self.layout()                                                   # sets self.N (an x0 given before any layout() crashed)
        r = least_squares(lambda x: self.residuals(x, idx), self.x0() if xi is None or len(xi) != self.N else np.asarray(xi),
                          method="trf", x_scale="jac", max_nfev=int(os.environ.get("FIT_NFEV", 300)))
        self.res = r
        return r.x

    def posterior(self, x, n, seed=0):
        """n Laplace draws of the parameter vector (J^T J of the residuals at x; pseudo-inverse on the null space)"""
        J = self.res.jac if hasattr(self, "res") else None
        if J is None or n <= 0: return [x] * max(n, 0)
        J = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
        H = J.T @ J; w, V = np.linalg.eigh(H)
        wi = np.where(w > 1e-8 * w.max(), 1.0 / np.maximum(w, 1e-300), 0.0)
        Lc = V * np.sqrt(wi)[None, :]
        rng = np.random.default_rng(seed)
        return [x + Lc @ rng.standard_normal(len(x)) for _ in range(n)]

    # ---- predictions for assignments that are not rows
    def tables(self, x):
        """per-tensor factors of the cost: s_w eb_w, the family log factor per (family, group), q; layer compounding"""
        u = self.unpack(x); D = self.D
        s = np.exp(self.A @ u["theta"] + u["beta"] * self.p)
        ph = {}
        for f in self.v3fams:
            j = self.v3fams.index(f); ph[f] = u["phi3"][f] + u["dev3"][j]
        emb = dict(zip(self.levels, u["emb"]))
        seen = sorted(int(l[1:]) for l in emb if l.startswith("e") and l[1:].isdigit())
        for j in range(len(D.emb_names)):                              # an embedding no row used: the nearest lower seen one's
            if f"e{j}" in emb or not seen: continue                    # offset (no predicted gain, no loss) -- the search measures it
            lower = [k for k in seen if k < j]; emb[f"e{j}"] = emb[f"e{lower[-1] if lower else seen[0]}"]
        return {"s": s, "eb": D.eb, "q": u["q"], "ph": ph, "kl": self.KA @ u["kap"], "c0": u["c0"],
                "emb": emb, "sat": u["sat"]}

    def cost(self, tb, i, err, fam):
        """one tensor's cost term at error err of family fam"""
        eb = tb["eb"][i]
        if eb <= 0: return 0.0
        g = self.D.grp[i] if self.D.grp[i] >= 0 else 0
        lph = tb["ph"][fam][g] if fam in tb["ph"] else 0.0
        return tb["s"][i] * eb * (err * math.exp(lph) / eb) ** tb["q"]

    def kl_from_X(self, tb, X, emb_label):
        P = tb["c0"] + tb["emb"].get(emb_label, 0.0)
        for l in range(len(X)): P = P * (1 + tb["kl"][l] * X[l]) + X[l]
        P = max(P, 1e-9)
        if tb["sat"]: P = tb["sat"] * -math.expm1(-P / tb["sat"])
        return P

    def layer_X(self, tb, a):
        X = np.zeros(self.D.NL + 1)
        for i, w in enumerate(self.D.W):
            o = self.D.M.opt(w, a[w]); X[self.D.layer[i]] += self.cost(tb, i, o["err"], o.get("fam", "v3"))
        return X

    def predict(self, x, a, tb=None):
        tb = tb or self.tables(x)
        return self.kl_from_X(tb, self.layer_X(tb, a), self.D.emb_label(a))

    def grad_X(self, tb, X, emb_label):
        """dP_final / dX_l (no saturation) by a backward pass"""
        P = [tb["c0"] + tb["emb"].get(emb_label, 0.0)]
        for l in range(len(X)): P.append(P[-1] * (1 + tb["kl"][l] * X[l]) + X[l])
        g = np.zeros(len(X)); acc = 1.0
        for l in range(len(X) - 1, -1, -1):
            g[l] = acc * (P[l] * tb["kl"][l] + 1); acc *= (1 + tb["kl"][l] * X[l])
        return g, P[-1]

    def dump(self, x):
        u = self.unpack(x); tb = self.tables(x)
        return {"model": "kfit3/search-v3", "x": x.tolist(), "q": u["q"], "c0": u["c0"], "kappa": tb["kl"].tolist(),
                "phi3": {k: math.exp(v) for k, v in u["phi3"].items()}, "slices": dict(zip(self.slices, u["soff"].tolist())),
                "emb": dict(zip(self.levels, u["emb"].tolist())), "beta": u["beta"],
                "tensors": [{"w": w, "s": float(tb["s"][i]), "eb": float(self.D.eb[i])} for i, w in enumerate(self.D.W)]}


def build(M, obs, opts=None):
    opts = opts or {}
    D = SData(M, obs)
    return SModel(D, tau=opts.get("tau", 0.02), phi_sd=opts.get("phi_sd", 1.0), beta=str(opts.get("beta", "0")),
                  theta_sd=opts.get("theta_sd", 4.0), v3_group_sd=opts.get("v3_group_sd", 0.3), smooth=opts.get("smooth", 0.5))


def fit_obs(M, obs, opts=None, x0=None, ndraw=0, seed=0):
    """-> (model, x, draws, report)"""
    Mo = build(M, obs, opts); x = Mo.fit(xi=x0)
    pr = Mo.predict_rows(x, list(range(len(Mo.D.rows)))); km = np.array([r["kld"] for r in Mo.D.rows])
    lr = np.log(pr / km); se = np.array([r["se"] for r in Mo.D.rows])
    rep = {"rows": len(obs), "labels": Mo.NS, "params": Mo.N, "rmse_log": float(np.sqrt(np.mean(lr ** 2))) if len(lr) else None,
           "chi2_per_row": float(np.mean(lr ** 2 / (se ** 2 + Mo.tau ** 2))) if len(lr) else None, "cost": float(Mo.res.cost), "nfev": int(Mo.res.nfev)}
    return Mo, x, Mo.posterior(x, ndraw, seed), rep


def check(M, obs, opts, test=(2.5, 4.5), main_min=2.0, frac=0.2, seed=0):
    """TODO 'Low sizes in parallel': does a joint fit (every size's rows) predict held-out main-size rows (sizes in
    [test]) better than a main-only fit (rows of sizes >= main_min)? Held out: a seeded fraction of the non-anchor rows
    (se > 0) of the test sizes; their labels keep their anchors, so each held-out row is predicted relative to its
    anchor. -> {arm: {"n", "logrmse"}}"""
    import random
    rng = random.Random(seed)
    test_rows = [o for o in obs if o.get("se", 0) > 0 and test[0] <= (o.get("size") or -1) <= test[1]]
    ho = {o["tag"] for o in test_rows if rng.random() < frac}
    out = {}
    for arm, keep in (("main-only", lambda o: (o.get("size") or 0) >= main_min), ("joint", lambda o: True)):
        tr = [o for o in obs if o["tag"] not in ho and keep(o)]
        Mo, x, _, _ = fit_obs(M, tr, opts)
        errs = []
        for o in obs:
            if o["tag"] not in ho: continue
            anc = [a for a in tr if a["label"] == o["label"] and a.get("se", 0) == 0]
            if not anc: continue
            a = anc[0]; lp = math.log(Mo.predict(x, o["assign"]) / Mo.predict(x, a["assign"])); lm = math.log(o["kld"] / a["kld"])
            errs.append(lp - lm)
        out[arm] = {"n": len(errs), "train_rows": len(tr), "logrmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else None}
    return out


def main(argv):
    if argv[0] == "check":                                            # fit.py check IN.json [SEED]
        d = json.load(open(argv[1])); M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
        r = check(M, d["obs"], d.get("opts"), seed=int(argv[2]) if len(argv) > 2 else 0); print(json.dumps(r)); return 0
    d = json.load(open(argv[0]))
    M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
    Mo, x, draws, rep = fit_obs(M, d["obs"], d.get("opts"), d.get("x0"), int(d.get("ndraw", 0)), int(d.get("seed", 0)))
    out = {"report": rep, "model": Mo.dump(x), "x": x.tolist(), "draws": [v.tolist() for v in draws], "obs_tags": [o["tag"] for o in d["obs"]]}
    json.dump(out, open(argv[1] + ".part", "w")); os.replace(argv[1] + ".part", argv[1])
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
