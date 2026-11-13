# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ev133.py -- #173 (w133): the RULES-173.md protocol (fit units, event predictions, metrics, bootstrap).

  python3 ev133.py fit  DS CANDS MODEL CUT OUTDIR [--noprior]   fit one model on rows with ri <= CUT (+ prior rows):
                                                                 OUTDIR/fit-MODEL-CUT[-np].json (skipped if present)
  python3 ev133.py pred DS CANDS MODEL CUT OUTDIR EVDS LO HI     predictions of EVDS's events with LO < ri <= HI
                                                                 (EVDS may be a later dataset: P) -> pred-...json
  python3 ev133.py s    DS CANDS MODEL OUTDIR                    S split (fit on S0 rows without prior; fit129 / s129)
  python3 ev133.py report OUTDIR SPLIT=pred-glob[,glob..] ...    metrics + paired bootstrap vs M0 -> stdout JSON
CANDS: {"models": {NAME: {"kind": "kfit", "ext": {...}, "opts": {...}} | {"kind": "resid", "base": NAME, "feats": [...],
"alpha": a | "cv"}}}. Single-threaded (OMP / BLAS 1), CPU, numpy + scipy."""
import gzip, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m133 as MM             # noqa: E402

MN, FT = MM.MN, MM.FT
TAU = 0.02
KB = [1.75, 2.5, 3.5, 4.5]                                             # K bands (option K <= edge)


def load(dsp):
    return json.load(gzip.open(dsp, "rt"))


class DS:
    def __init__(self, dsp):
        self.d = d = load(dsp)
        self.M = MN.Menu.load(os.environ.get("MENU", d["menu"]))
        self.W = d["W"]; self.A = d["assign"]; self._c = {}

    def asg(self, t):
        if t not in self._c:
            self._c[t] = {w: self.M.T[self.M.WI[w]]["opts"][j]["o"] for w, j in zip(self.W, self.A[t])}
        return self._c[t]

    def obs(self, cut, prior=True, relabel=None):
        """fit rows; relabel "size": a search row's label gets its size (S@size|p|..): one offset per (label, size)"""
        R = [o for o in self.d["obs"] if o["ri"] <= cut] + (self.d["prior"] if prior else [])
        lab = lambda o: (o["label"].replace("|", f"@{o['size']:g}|", 1) if relabel == "size" and o["ri"] > 0 else o["label"])
        return [dict(tag=o["tag"], assign=self.asg(o["a"]), label=lab(o), kld=o["kld"], se=o["se"], mode=o.get("mode", "f16"))
                for o in R]

    def events(self, lo, hi):
        return [e for e in self.d["events"] if lo < e["ri"] <= hi]


def fitpath(out, name, cut, noprior=False): return f"{out}/fit-{name}-{cut}{'-np' if noprior else ''}.json"


def get_model(ds, cands, name, cut, out, noprior=False, refit=True):
    """(model object, x) of a kfit-kind model, fitted or loaded"""
    c = cands["models"][name]
    obs = ds.obs(cut, prior=not noprior, relabel=c.get("relabel"))
    Mo = MM.build(ds.M, obs, c.get("opts", {"tau": 0.02, "beta": "0", "xscale": True}), c.get("ext"))
    fp = fitpath(out, name, cut, noprior)
    if os.path.exists(fp):
        x = np.array(json.load(open(fp))["x"]); Mo.layout(); return Mo, x
    if not refit: raise FileNotFoundError(fp)
    t0 = time.time(); x = Mo.fit()
    pr = Mo.predict_rows(x, list(range(len(obs)))); km = np.array([o["kld"] for o in obs])
    rep = {"rows": len(obs), "params": int(Mo.N), "cost": float(Mo.res.cost), "nfev": int(Mo.res.nfev), "secs": round(time.time() - t0, 1),
           "rmse_log": float(np.sqrt(np.mean(np.log(pr / km) ** 2))), "lam": Mo.lam(x)}
    json.dump({"model": name, "cut": cut, "noprior": noprior, "report": rep, "x": x.tolist()}, open(fp + ".part", "w"))
    os.replace(fp + ".part", fp)
    print(f"fit {name} cut {cut}{' noprior' if noprior else ''}: {rep}", flush=True)
    return Mo, x


def kfit_pred(Mo, x, ds_ev, evs):
    """per event, per part: model log ratio candidate / incumbent"""
    tags = sorted({e["tag"] for e in evs} | {e["ref"] for e in evs}); ti = {t: i for i, t in enumerate(tags)}
    out = {}
    for p in ("c", "x"):
        modes = {e["mode"] for e in evs}
        lp = {}
        for m in modes:
            lp[m] = Mo.logpart(x, [ds_ev.asg(t) for t in tags], p, m)
        for e in evs:
            v = lp[e["mode"]]; out.setdefault(e["tag"], {})[p] = float(v[ti[e["tag"]]] - v[ti[e["ref"]]])
    return out


# ------------------------------------------------------------------------------------- M2: residual ridge on a base
def lin_table(Mo, x):
    """moves.Pred over the model (per-tensor cost table) -- for the linearised per-tensor contributions"""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
    import moves as MV
    return MV.Pred(Mo, x)


def rfeat(Mo, PR, a_ref, a_c, feats):
    """features of a move ref -> candidate: per-tensor linearised d log P under the base (at ref), summed by cell"""
    D = Mo.D; M = D.M
    J0, dl = PR.lin(a_ref); J1 = PR.idx(a_c)
    v = np.array([dl[i][J1[i]] for i in range(len(J1))])                 # per tensor: linearised d log P of the move
    nl = D.NL; G = Mo.GH
    out = []
    for f in feats:
        if f == "gb8":                                                  # group x 8-layer band (+ head)
            z = np.zeros(len(GL) * 8 + 1)
            for i in range(len(v)):
                if v[i] == 0: continue
                z[len(GL) * 8 if G[i] == len(GL) else G[i] * 8 + min(D.layer[i] // 8, 7)] += v[i]
            out.append(z)
        elif f == "gk":                                                 # group x K band of the lower option (+ head)
            z = np.zeros(len(GL) * (len(KB) + 1) + 1)
            for i, w in enumerate(D.W):
                if v[i] == 0: continue
                if G[i] == len(GL): z[-1] += v[i]; continue
                t = M.T[M.WI[w]]["opts"]; k = min(t[J0[i]]["K"] or 0, t[J1[i]]["K"] or 0)
                z[G[i] * (len(KB) + 1) + int(np.searchsorted(KB, k))] += v[i]
            out.append(z)
        elif f == "hbar":                                               # x log hbar (centred per group, as fit.py's p)
            z = np.zeros(len(GL))
            for i in range(len(v)):
                if v[i] and G[i] < len(GL): z[G[i]] += v[i] * Mo.p[i]
            out.append(z)
        elif f == "tot":                                                # the total (a global slope correction)
            out.append(np.array([v.sum()]))
    return np.concatenate(out)


def ridge(X, y, w, alpha):
    Xw = X * np.sqrt(w)[:, None]; yw = y * np.sqrt(w)
    return np.linalg.solve(Xw.T @ Xw + alpha * np.eye(X.shape[1]), Xw.T @ yw)


def wmse(y, p, w): return float(np.sum(w * (y - p) ** 2) / np.sum(w))


def resid_fit(ds, cands, name, cut, out):
    c = cands["models"][name]; fp = fitpath(out, name, cut)
    Mo, x = get_model(ds, cands, c["base"], cut, out)
    if os.path.exists(fp): return Mo, x, json.load(open(fp))
    PR = lin_table(Mo, x)
    evs = ds.events(0, cut); base = kfit_pred(Mo, x, ds, evs)
    F = np.array([rfeat(Mo, PR, ds.asg(e["ref"]), ds.asg(e["tag"]), c["feats"]) for e in evs])
    res = {}
    for p in ("c", "x"):
        ix = [j for j, e in enumerate(evs) if p in e["parts"] and e["parts"][p]["se"] is not None]
        lam = Mo.lam(x) if p == "x" else 1.0                           # x part: the feature is on log P, the target on x
        X = F[ix] * lam; y = np.array([evs[j]["parts"][p]["logr"] - base[evs[j]["tag"]][p] for j in ix])
        w = np.array([1 / (evs[j]["parts"][p]["se"] ** 2 + TAU ** 2) for j in ix])
        al = c.get("alpha", "cv"); cvr = None
        if al == "cv":                                                  # 5 contiguous time blocks
            grid = [0.3, 1, 3, 10, 30, 100, 300, 1000]; fold = np.array_split(np.arange(len(ix)), 5); cvr = {}
            for a_ in grid:
                err = []
                for f in fold:
                    tr = np.setdiff1d(np.arange(len(ix)), f); b = ridge(X[tr], y[tr], w[tr] / w.mean(), a_)
                    err.append(np.sum(w[f] * (y[f] - X[f] @ b) ** 2))
                cvr[a_] = float(np.sum(err) / w.sum())
            cvr[0] = wmse(y, 0 * y, w)
            al = min(grid, key=lambda a_: cvr[a_])
        b = ridge(X, y, w / w.mean(), float(al))
        res[p] = {"beta": b.tolist(), "alpha": al, "cv": cvr, "n": len(ix), "in_wmse": wmse(y, X @ b, w), "in_wmse0": wmse(y, 0 * y, w)}
    json.dump({"model": name, "cut": cut, "res": res}, open(fp + ".part", "w")); os.replace(fp + ".part", fp)
    print(f"fit {name} cut {cut}: " + json.dumps({p: {k: v for k, v in r.items() if k not in ('beta', 'cv')} for p, r in res.items()}), flush=True)
    return Mo, x, json.load(open(fp))


def predict(ds, cands, name, cut, out, ds_ev, evs, noprior=False):
    c = cands["models"][name]
    if c["kind"] == "kfit":
        Mo, x = get_model(ds, cands, name, cut, out, noprior); return kfit_pred(Mo, x, ds_ev, evs), int(Mo.N)
    Mo, x, R = resid_fit(ds, cands, name, cut, out)
    base = kfit_pred(Mo, x, ds_ev, evs); PR = lin_table(Mo, x)
    for e in evs:
        f = rfeat(Mo, PR, ds_ev.asg(e["ref"]), ds_ev.asg(e["tag"]), c["feats"])
        for p in ("c", "x"):
            base[e["tag"]][p] += float(f @ np.array(R["res"][p]["beta"])) * (Mo.lam(x) if p == "x" else 1.0)
    return base, int(Mo.N) + sum(len(R["res"][p]["beta"]) for p in ("c", "x"))


# ------------------------------------------------------------------------------------- metrics
def metrics(evs, P, B=0, seed=173, base=None):
    """RULES-173 metrics of predictions P ({tag: {c, x}}) on events; base: M0's P for paired differences"""
    def one(ix, PP):
        r = {}
        tot = []
        for p in ("c", "x"):
            y, pr, w, se = [], [], [], []
            for j in ix:
                e = evs[j]; v = e["parts"].get(p)
                if not v or v["se"] is None: continue
                y.append(v["logr"]); pr.append(PP[e["tag"]][p]); se.append(v["se"]); w.append(1 / (v["se"] ** 2 + TAU ** 2))
            y, pr, w, se = map(np.array, (y, pr, w, se))
            if not len(y): continue
            mse = float(np.mean((y - pr) ** 2))
            r[p] = {"n": len(y), "wmse": wmse(y, pr, w), "rmse": math.sqrt(mse), "rmse_nc": math.sqrt(max(0.0, mse - float(np.mean(se ** 2)))),
                    "slope": float(np.sum(w * (pr - np.average(pr, weights=w)) * (y - np.average(y, weights=w))) /
                                   max(np.sum(w * (pr - np.average(pr, weights=w)) ** 2), 1e-30))}
            tot.append(r[p]["wmse"])
        r["wmse"] = float(np.mean(tot)) if tot else None
        # calls
        pa, ma, sc, cl = [], [], [], []
        for j in ix:
            e = evs[j]; pp = [PP[e["tag"]][p] for p in e["parts"] if e["parts"][p]["se"] is not None] or [PP[e["tag"]][p] for p in e["parts"]]
            st = float(np.mean(pp)) + e["off"]
            pa.append(st < 0 and all(v + e["off"] < 0 for v in pp)); ma.append(e["decision"] == "accept"); sc.append(-st)
            mv = [v for v in e["parts"].values() if v["se"] is not None]
            if mv:
                ms = float(np.mean([v["logr"] for v in mv])) + e["off"]; mse_ = math.sqrt(sum(v["se"] ** 2 for v in mv)) / len(mv)
                if abs(ms) > 2 * mse_: cl.append((st < 0) == (ms < 0))
        pa, ma, sc = np.array(pa), np.array(ma), np.array(sc)
        rec_a = float(np.mean(pa[ma])) if ma.any() else float("nan"); rec_r = float(np.mean(~pa[~ma])) if (~ma).any() else float("nan")
        r["bal_acc"] = 0.5 * (rec_a + rec_r); r["acc"] = float(np.mean(pa == ma)); r["acc_recall"] = rec_a; r["rej_recall"] = rec_r
        if ma.any() and (~ma).any():
            sa, sr = sc[ma], sc[~ma]; r["auc"] = float(np.mean((sa[:, None] > sr[None, :]) + 0.5 * (sa[:, None] == sr[None, :])))
        r["clear_sign"] = float(np.mean(cl)) if cl else None; r["n_clear"] = len(cl)
        acc_ix = [j for j in ix if evs[j]["decision"] == "accept"]
        if acc_ix:
            pg = [np.mean([PP[evs[j]["tag"]][p] for p in evs[j]["parts"]]) for j in acc_ix]
            mg = [np.mean([v["logr"] for v in evs[j]["parts"].values()]) for j in acc_ix]
            r["acc_pred"] = float(np.mean(pg)); r["acc_meas"] = float(np.mean(mg)); r["n_acc"] = len(acc_ix)
        r["n"] = len(ix)
        return r
    ix = list(range(len(evs))); out = {"point": one(ix, P)}
    if base is not None: b0 = one(ix, base); out["d_point"] = {"wmse": out["point"]["wmse"] - b0["wmse"],
                                                               "rel": out["point"]["wmse"] / b0["wmse"] - 1,
                                                               "bal_acc": out["point"]["bal_acc"] - b0["bal_acc"]}
    if B:
        rng = np.random.default_rng(seed); bs = {"wmse": [], "c": [], "x": [], "bal_acc": [], "auc": [], "d_wmse": [], "d_rel": [], "d_bal": []}
        for _ in range(B):
            s = list(rng.integers(0, len(evs), len(evs))); m = one(s, P)
            bs["wmse"].append(m["wmse"]); bs["c"].append(m.get("c", {}).get("wmse", np.nan)); bs["x"].append(m.get("x", {}).get("wmse", np.nan))
            bs["bal_acc"].append(m["bal_acc"]); bs["auc"].append(m.get("auc", np.nan))
            if base is not None:
                m0 = one(s, base); bs["d_wmse"].append(m["wmse"] - m0["wmse"]); bs["d_rel"].append(m["wmse"] / m0["wmse"] - 1)
                bs["d_bal"].append(m["bal_acc"] - m0["bal_acc"])
        out["ci"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items() if v}
    return out


def main(argv):
    cmd = argv[0]
    if cmd in ("fit", "pred", "s"):
        ds = DS(argv[1]); cands = json.load(open(argv[2])); name = argv[3]
    if cmd == "fit":
        cut, out = int(argv[4]), argv[5]; os.makedirs(out, exist_ok=True)
        if cands["models"][name]["kind"] == "kfit": get_model(ds, cands, name, cut, out, noprior="--noprior" in argv)
        else: resid_fit(ds, cands, name, cut, out)
        return 0
    if cmd == "pred":
        cut, out, evp, lo, hi = int(argv[4]), argv[5], argv[6], int(argv[7]), int(argv[8])
        dse = ds if evp == argv[1] else DS(evp)
        evs = dse.events(lo, hi); P, npar = predict(ds, cands, name, cut, out, dse, evs)
        tagx = os.path.basename(evp).split(".")[0]
        json.dump({"model": name, "cut": cut, "lo": lo, "hi": hi, "evds": evp, "params": npar, "pred": P},
                  open(f"{out}/pred-{name}-{cut}-{tagx}-{lo}-{hi}.json", "w"))
        print(f"pred {name} cut {cut} on {len(evs)} events of {tagx} ({lo}, {hi}]", flush=True); return 0
    if cmd == "s":                                                      # S split
        out = argv[4]; cut = ds.d["n_results"]
        c = cands["models"][name]
        if c["kind"] != "kfit": print("S split: kfit-kind models only (a residual model needs events, which S has none of)"); return 0
        Mo, x = get_model(ds, cands, name, cut, out, noprior=True)
        pri = ds.d["prior"]; pa = [ds.asg(o["a"]) for o in pri]
        lp = Mo.logpart(x, pa, "c"); km = np.log([o["kld"] for o in pri]); err = lp - km; err = err - err.mean()
        s = {"fit129_within_rmse": float(np.sqrt(np.mean(err ** 2))), "n": len(pri)}
        e129 = {}
        for r in ds.d["s129"]:
            for p in ("c", "x"):
                v = Mo.logpart(x, [ds.asg(r["tag"]), ds.asg(r["ref"])], p); e129.setdefault(p, []).append(float(v[0] - v[1] - r["parts"][p]["logr"]))
        s["s129_rmse"] = {p: float(np.sqrt(np.mean(np.square(v)))) for p, v in e129.items()}
        s["s129_mean"] = float(np.mean(list(s["s129_rmse"].values())))
        json.dump(s, open(f"{out}/S-{name}.json", "w")); print(f"S {name}: {s}", flush=True); return 0
    if cmd == "report":
        out = argv[1]; B = int(os.environ.get("BOOT", 4000)); res = {}
        for spec in argv[2:]:
            split, pat = spec.split("=", 1)
            import glob
            files = sorted(glob.glob(f"{out}/{pat}"))
            by = {}
            for f in files:
                d = json.load(open(f)); by.setdefault(d["model"], []).append(d)
            evds = {}
            def evs_of(d):
                if d["evds"] not in evds: evds[d["evds"]] = DS(d["evds"])
                return [e for e in evds[d["evds"]].events(d["lo"], d["hi"])]
            # pooled events across the files of one model (same files per model by construction)
            pooled = {}
            for m, L in by.items():
                E, P = [], {}
                for d in sorted(L, key=lambda d: (d["evds"], d["lo"])):
                    for e in evs_of(d):
                        if e["tag"] in d["pred"]: E.append(e); P[e["tag"]] = d["pred"][e["tag"]]
                pooled[m] = (E, P, L[0]["params"])
            mode = os.environ.get("MODE")
            if mode: pooled = {m: ([e for e in E if e["mode"] == mode], P, n) for m, (E, P, n) in pooled.items()}
            base = pooled.get("M0")
            res[split] = {}
            for m, (E, P, npar) in pooled.items():
                r = metrics(E, P, B=B, base=base[1] if base and m != "M0" else None)
                r["params"] = npar; res[split][m] = r
                pt = r["point"]
                print(f"{split:6s} {m:14s} n {pt['n']:4d} par {npar:4d} wMSE {pt['wmse']:.5f} c {pt.get('c', {}).get('wmse', float('nan')):.5f} "
                      f"x {pt.get('x', {}).get('wmse', float('nan')):.5f} | rmse c {pt.get('c', {}).get('rmse', float('nan')):.4f} x {pt.get('x', {}).get('rmse', float('nan')):.4f} "
                      f"| bal {pt['bal_acc']:.3f} auc {pt.get('auc', float('nan')):.3f} clear {pt['clear_sign']} | slope c {pt.get('c', {}).get('slope', float('nan')):.2f} "
                      f"x {pt.get('x', {}).get('slope', float('nan')):.2f}"
                      + (f" | d {100 * r['d_point']['rel']:+.1f}% CI {[round(100 * v, 1) for v in r['ci']['d_rel']]}" if 'd_point' in r and 'ci' in r else ""), flush=True)
        json.dump(res, sys.stdout if os.environ.get("JSON") else open(os.devnull, "w"), indent=1)
        if os.environ.get("REPORT"): json.dump(res, open(os.environ["REPORT"], "w"), indent=1)
        return 0


GL = FT.GL

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
