#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""floor129.py SPECS RESULTS DOCSDIR MENU OUTDIR -- #169 follow-up (report only, after the rule's verdict): why the
held-out s129 ratios miss (H log-RMSE ~0.13, the xcorp part ~0.14 vs calib ~0.03), and a candidate fix for fit.py.
Hypothesis: a slice whose KL has a large part the trunk does not move (xcorp: head / embedding / corpus floor, KLD 3.6x
calib's at the same trunk) compresses trunk-swap ratios; fit.py's per-label offset is multiplicative and cannot carry
it. Arms (β 0, trained on U + G on fit129 + the two s129 anchors u-k2.5 / u-k4, predicting the 8 H swaps' ratio to their
base on s129):
  A pooled s129 label, no floor (fit129.py's H)   B pooled, + an additive floor per label (fit129's floor fixed at 0)
  C parts as their own labels (s129x, s129c)      D parts + floor
Also: the final refit on fit129 rows only (U + G + P), β 0 -> OUTDIR/fit129c.out.json (fit.py output format)."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import fit as FT      # noqa: E402
import fit129 as F9   # noqa: E402
import menu as MN     # noqa: E402
from scipy.optimize import least_squares  # noqa: E402


class FModel(FT.SModel):
    """fit.py's model + per-label additive floor: pred = P exp(soff) + f_label (f of the first label = 0)"""
    def layout(self):
        L = super().layout(); n = self.N
        L["floor"] = slice(n, n + self.NS - 1); self.N = n + self.NS - 1
        return L

    def predict_rows(self, x, idx):
        L = self.layout(); idx = np.asarray(idx, int)
        f = np.r_[0.0, np.exp(x[L["floor"]])]
        return super().predict_rows(x, idx) + f[self.SLI[idx]]

    def x0(self):
        x = super().x0(); L = self.layout()
        if self.NS > 1: x[L["floor"]] = math.log(0.02)
        return x

    def residuals(self, x, idx):
        L = self.layout(); r = super().residuals(x, idx)
        return np.r_[r, (x[L["floor"]] - math.log(0.02)) / 4.0] if self.NS > 1 else r


def build(M, obs, floor):
    D = FT.SData(M, obs)
    cls = FModel if floor else FT.SModel
    return cls(D, tau=0.02, phi_sd=1.0, beta="0", theta_sd=4.0, v3_group_sd=0.3, smooth=0.5)


def fit_idx(Mo, idx):
    r = least_squares(lambda x: Mo.residuals(x, idx), Mo.x0(), method="trf", x_scale="jac", max_nfev=300)
    Mo.res = r; return r.x


def main(specs, results, docsdir, menu, outdir):
    S, R, D = F9.load_rows(specs, results, docsdir); M = MN.Menu.load(menu)
    a = lambda t: F9.assign_of(S[t])
    fitrows = [{"tag": f"{t}@fit129", "assign": a(t), "label": "fit129", "kld": r["kld"], "se": 0.0, "kind": S[t]["w129"]["kind"]}
               for (t, sl), r in R.items() if sl == "fit129"]
    H = [t for (t, sl) in R if sl == "s129" and S[t]["w129"]["kind"] == "G"]
    rep = {}
    for arm, parts, floor in (("A", False, False), ("B", False, True), ("C", True, False), ("D", True, True)):
        rows = [o for o in fitrows if o["kind"] in ("U", "G")]
        held = []
        for t in ["w129-u-k2.5", "w129-u-k4"] + H:
            r = R[(t, "s129")]
            labs = [("s129x", r["parts"]["x"]["kld"]), ("s129c", r["parts"]["c"]["kld"])] if parts else [("s129", r["kld"])]
            for lab, k in labs:
                o = {"tag": f"{t}@{lab}", "assign": a(t), "label": lab, "kld": k, "se": 0.0}
                (held if t in H else rows).append(o)
        Mo = build(M, rows + held, floor); n = len(rows); x = fit_idx(Mo, list(range(n)))
        P = Mo.predict_rows(x, list(range(len(rows) + len(held)))); by = {o["tag"]: (P[i], o["kld"]) for i, o in enumerate(rows + held)}
        errs = {}
        for o in held:
            t, lab = o["tag"].split("@"); bt = f"w129-u-k{S[t]['w129']['base']:g}@{lab}"
            pr, me = by[o["tag"]]; pb, mb = by[bt]
            errs[o["tag"]] = {"pred": math.log(pr / pb), "meas": math.log(me / mb)}
        e = np.array([v["pred"] - v["meas"] for v in errs.values()])
        L = Mo.layout(); u = Mo.unpack(x)
        fl = dict(zip(Mo.slices[1:], np.exp(x[L["floor"]]).tolist())) if floor else {}
        rep[arm] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "by_label": {lab: float(np.sqrt(np.mean([(v["pred"] - v["meas"]) ** 2 for k, v in errs.items() if k.endswith(lab)])))
                    for lab in {k.split("@")[1] for k in errs}}, "floors": fl, "offsets": dict(zip(Mo.slices, u["soff"].tolist())), "rows": errs}
        print(f"[{arm}] parts {parts} floor {floor}: H log-RMSE {rep[arm]['rmse']:.4f} by label {({k: round(v, 4) for k, v in rep[arm]['by_label'].items()})} floors {({k: round(v, 5) for k, v in fl.items()})}")
    # the fit129-only final refit (β 0)
    Mo, x, draws, frep = FT.fit_obs(M, fitrows, {"beta": "0"}, ndraw=16, seed=169)
    out = {"report": frep, "model": Mo.dump(x), "x": x.tolist(), "draws": [v.tolist() for v in draws], "obs_tags": [o["tag"] for o in fitrows],
           "w129": {"menu": menu, "beta_mode": "0", "rows": "fit129 only (U + G + P)"}}
    json.dump(out, open(f"{outdir}/fit129c.out.json", "w")); rep["fit129c"] = frep
    print(f"fit129-only refit (beta 0): {frep}")
    json.dump(rep, open(f"{outdir}/floor129.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
