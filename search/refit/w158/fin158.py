#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fin158.py -- (w158, #173-03, RULES-173-03) fit / CV / curve / re-solve with m158's M1m on the mad menu. CPU (colo).

  fin158.py fit   OUT.json MENU MODELS.json NAME --obs O.jsonl.. [--drop-block B,..]       fit on every row (or without blocks)
  fin158.py cv    OUT.json MENU MODELS.json NAME --obs .. [--on BLOCKS] [--k 5]            k-fold CV (rows sorted by tag, i mod k;
                  --on: only rows of these blocks are held out, the rest always train): weighted MSE of log KLD
                  (weights 1 / (SE^2 + 0.02^2), the profiled offset from the training rows), and the sign agreement of
                  held-out swap rows' predicted vs measured log ratio vs their base row (|measured| > 2 SE)
  fin158.py pred  OUT.json MENU MODELS.json NAME FIT.json --obs ..                          predictions of the given rows
  fin158.py curve OUT.json MENU MODELS.json NAME FIT.json --obs ..                          uniform-level KL (head u6.5, emb t8m)
  fin158.py solve OUT.json MENU MODELS.json NAME FIT.json --obs .. --sizes 1,1.25,.. [--min-k K] [--save-dir D] [--tag T]
                  per size: mckp from the uniform level below the size (head / embedding nearest K + 0.5) and from the
                  next lower size's pick; window [size - 0.02, size] in container bpw (a/b T5 bytes added at >= 5.5)
MODELS.json: {"NAME": {"ext": {...}, "opts": {...}}}; obs lines: run158's collect output (tag, assign, label, mode, kld, se,
block, base_tag)."""
import argparse, collections, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m158 as M8             # noqa: E402
import menu as MN             # noqa: E402
import moves as MV            # noqa: E402

TAU = 0.02


def load_obs(paths, drop=()):
    out = []
    for p in paths:
        for l in open(p):
            if l.strip():
                o = json.loads(l)
                if o.get("block") not in drop: out.append(o)
    return out


def build(M, obs, spec):
    return M8.build(M, obs, spec.get("opts", {"tau": TAU, "beta": "0"}), spec.get("ext"))


def fit_rows(M, obs, spec, idx=None):
    tr = obs if idx is None else [obs[i] for i in idx]
    Mo = build(M, tr, spec); x = Mo.fit(); return Mo, x


def predict_for(Mo, x, obs_tr, rows):
    """log predictions (profiled offset from the training rows) for arbitrary rows"""
    def lpart(rs):                                                  # log prediction without the offset, per row's act mode
        return np.array([float(Mo.logpart(x, [o["assign"]], "c", mode=M8.mmode(o))[0]) for o in rs])
    g_tr = lpart(obs_tr); km = np.log([o["kld"] for o in obs_tr]); w = 1 / (np.array([o["se"] for o in obs_tr]) ** 2 + TAU ** 2)
    off = float(np.sum(w * (km - g_tr)) / np.sum(w))
    return lpart(rows) + off


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("out"); ap.add_argument("menu"); ap.add_argument("models")
    ap.add_argument("name"); ap.add_argument("fitf", nargs="?"); ap.add_argument("--obs", nargs="+", required=True)
    ap.add_argument("--drop-block", default=""); ap.add_argument("--on", default=""); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sizes"); ap.add_argument("--min-k", type=float, default=None); ap.add_argument("--save-dir"); ap.add_argument("--tag", default="pick158")
    ap.add_argument("--tol", type=float, default=0.02); ap.add_argument("--head0", default="u6.5"); ap.add_argument("--emb0", default="t8m")
    a = ap.parse_args(argv)
    M = MN.Menu.load(a.menu); spec = json.load(open(a.models))[a.name]
    obs = load_obs(a.obs, set(filter(None, a.drop_block.split(","))))
    if a.cmd == "fit":
        Mo, x = fit_rows(M, obs, spec)
        pr = Mo.predict_rows(x, list(range(len(obs)))); km = np.array([o["kld"] for o in obs]); se = np.array([o["se"] for o in obs])
        lr = np.log(pr / km)
        u = Mo.unpack(x)
        rep = {"model": a.name, "rows": len(obs), "params": int(Mo.N), "cost": float(Mo.res.cost),
               "rmse_log": float(np.sqrt(np.mean(lr ** 2))), "wsse": float(np.sum(lr ** 2 / (se ** 2 + TAU ** 2))),
               "wsse_by_block": {b: float(np.sum([lr[i] ** 2 / (se[i] ** 2 + TAU ** 2) for i in range(len(obs)) if obs[i].get("block") == b]))
                                 for b in sorted({o.get("block") for o in obs})},
               "q": float(u["q"]), "c0": float(u["c0"]), "kappa_knots": [float(v) for v in u["kap"]], "eta": u.get("eta"),
               "lam1_mu1": [float(x[Mo.layout()["lam1"]][0]), float(x[Mo.layout()["mu1"]][0])] if "lam1" in Mo.layout() else None,
               "mode": [float(v) for v in x[Mo.layout()["mo"]]] if "mo" in Mo.layout() else None, "sat": u.get("sat"),
               "resid": {o["tag"]: round(float(v), 4) for o, v in zip(obs, lr)}}
        json.dump({"model": a.name, "spec": spec, "menu": a.menu, "obs_tags": [o["tag"] for o in obs], "report": rep, "x": x.tolist()}, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "resid"})); return 0
    if a.cmd == "cv":
        on = set(filter(None, a.on.split(",")))
        order = sorted(range(len(obs)), key=lambda i: obs[i]["tag"])
        cand = [i for i in order if not on or obs[i].get("block") in on]
        folds = [[i for j, i in enumerate(cand) if j % a.k == f] for f in range(a.k)]
        bytag = {o["tag"]: i for i, o in enumerate(obs)}
        res, sg = [], []
        for f, ho in enumerate(folds):
            if not ho: continue
            tr = [i for i in range(len(obs)) if i not in set(ho)]
            Mo, x = fit_rows(M, obs, spec, tr)
            lp = predict_for(Mo, x, [obs[i] for i in tr], [obs[i] for i in ho])
            for i, l in zip(ho, lp):
                o = obs[i]; e = l - math.log(o["kld"]); w = 1 / (o["se"] ** 2 + TAU ** 2); res.append((o["tag"], e, w, o.get("block")))
                b = o.get("base_tag")
                if b and b in bytag:
                    ob = obs[bytag[b]]; lb = predict_for(Mo, x, [obs[j] for j in tr], [ob])[0]
                    mr = math.log(o["kld"] / ob["kld"]); pr = l - lb
                    if abs(mr) > 2 * max(o.get("se_vs_base", o["se"]), 1e-9): sg.append(int(np.sign(mr) == np.sign(pr)))
            print(f"fold {f}: {len(ho)} held out", flush=True)
        E = np.array([r[1] for r in res]); Wt = np.array([r[2] for r in res])
        rep = {"model": a.name, "k": a.k, "on": sorted(on), "n": len(res), "wmse": float(np.sum(Wt * E ** 2) / np.sum(Wt)),
               "rmse": float(np.sqrt(np.mean(E ** 2))), "sign_n": len(sg), "sign_acc": float(np.mean(sg)) if sg else None,
               "by_block": {b: float(np.sqrt(np.mean([r[1] ** 2 for r in res if r[3] == b]))) for b in sorted({r[3] for r in res})},
               "rows": [{"tag": r[0], "err": round(r[1], 4), "block": r[3]} for r in res]}
        json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in rep.items() if k != "rows"})); return 0
    F = json.load(open(a.fitf)); Mo = build(M, obs, spec); Mo.layout(); x = np.array(F["x"]); assert len(x) == Mo.N, (len(x), Mo.N)
    g_tr = np.array([float(Mo.logpart(x, [o["assign"]], "c", mode=M8.mmode(o))[0]) for o in obs])
    w_tr = 1 / (np.array([o["se"] for o in obs]) ** 2 + TAU ** 2)
    OFF = float(np.sum(w_tr * (np.log([o["kld"] for o in obs]) - g_tr)) / np.sum(w_tr))
    klP = lambda P: float(math.exp(OFF + Mo.gpart(x, np.array([math.log(P)]), np.array([False]))[0]))   # A8 calib-scale KL of a P
    if a.cmd == "pred":
        lp = predict_for(Mo, x, obs, obs)
        json.dump({o["tag"]: {"pred": float(math.exp(l)), "kld": o["kld"]} for o, l in zip(obs, lp)}, open(a.out, "w"), indent=1); return 0
    if a.cmd == "curve":
        out = {}; PRc = MV.Pred(Mo, x)
        for lv in M.levels():
            u = M.uniform(lv, a.head0, a.emb0)
            out[lv] = {"K": M.opt(M.W[0], lv)["K"], "bpw": M.bpw(u), "pred": float(math.exp(OFF + Mo.logpart(x, [u], "c", mode="f16")[0])),
                       "pred_knap": klP(PRc.kl(u))}                         # the knapsack's path (tables): must equal pred
            print(f"{lv:28s} {out[lv]['bpw']:.3f} bpw  pred {out[lv]['pred']:.5f}")
        ks = [v["pred"] for v in out.values()]
        out["_monotone"] = all(b < a_ for a_, b in zip(ks, ks[1:]))
        print("strictly decreasing:", out["_monotone"])
        json.dump(out, open(a.out, "w"), indent=1); return 0
    # solve
    PR = MV.Pred(Mo, x)
    if a.min_k is not None:
        bad = [np.array([(o.get("K") is not None and t["w"] not in (MN.HEAD, MN.EMB) and o["K"] < a.min_k) for o in t["opts"]])
               for t in (M.T[M.WI[w]] for w in Mo.D.W)]
        lin0 = PR.lin
        def lin(asg):
            J, dl = lin0(asg)
            return J, [np.where(b, 10.0, d) for b, d in zip(bad, dl)]
        PR.lin = lin
    fixed0 = M.fixed; ext = M.d.get("ab_t5_extra") or 0
    hk = {o["o"]: o["K"] for o in M.T[M.WI[MN.HEAD]]["opts"]}; ek = {o["o"]: o["K"] for o in M.T[M.WI[MN.EMB]]["opts"]}
    near = lambda d, k: min(d, key=lambda n: (abs(d[n] - k), n))
    lv = M.levels(); res = {"model": a.name, "fit": a.fitf, "menu": a.menu, "min_k": a.min_k, "sizes": {}}; prev = None
    for size in [float(s) for s in a.sizes.split(",")]:
        M.fixed = fixed0 + (ext if size >= 5.5 else 0)
        hi = size * M.P / 8; lo = (size - a.tol) * M.P / 8
        starts = {}
        under = []
        for n in lv:
            K = M.opt(M.W[0], n)["K"]
            if a.min_k is not None and K < a.min_k: continue
            u = M.uniform(n, near(hk, K + 0.5), near(ek, K + 0.5))
            if M.bytes(u) <= hi: under.append(u)
        if under: starts["uniform"] = under[-1]
        else: starts["uniform"] = M.uniform(lv[0], min(hk, key=hk.get), min(ek, key=ek.get))
        if prev is not None: starts["prev"] = dict(prev)
        best = None; per = {}
        for k, s0 in starts.items():
            p = MV.mckp(PR, s0, hi, lo)
            if p is None: continue
            kl = klP(PR.kl(p)); inw = lo <= M.bytes(p) <= hi; per[k] = {"kl": kl, "inwin": inw, "bpw": M.bpw(p)}
            if best is None or (inw, -kl) > (best[2], -best[0]): best = (kl, p, inw, k)
        if best is None or not best[2]:
            res["sizes"][f"{size:g}"] = {"infeasible": True, "starts": per}; print(f"size {size:g}: no pick in the window", flush=True); continue
        pick = best[1]; prev = pick
        hist = collections.Counter(M.opt(w, pick[w])["K"] for w in M.W if w not in (MN.HEAD, MN.EMB))
        u0 = starts["uniform"]; ch = [w for w in pick if pick[w] != u0.get(w)]
        mb = sum(abs(M.opt(w, pick[w])["bytes"] - M.opt(w, u0[w])["bytes"]) for w in ch)
        tb = (M.bytes(pick) - M.opt(MN.EMB, pick[MN.EMB])["bytes"]); tp = M.P - 248320 * 5120
        r = {"from": best[3], "bpw": M.bpw(pick), "bpw_no_emb": tb * 8 / tp, "pred_kl": best[0], "head": pick[MN.HEAD], "emb": pick[MN.EMB],
             "ab": "t5" if size >= 5.5 else "t4", "levels": {f"{k:g}": v for k, v in sorted(hist.items())}, "starts": per,
             "vs_uniform": {"changed": len(ch), "moved_MB": round(mb / 1e6, 1), "uniform_kl": klP(PR.kl(u0)), "uniform_bpw": M.bpw(u0)}}
        res["sizes"][f"{size:g}"] = r
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True); json.dump(pick, open(f"{a.save_dir}/{a.tag}-{size:g}.json", "w"))
        print(f"size {size:g}: {r['bpw']:.4f} bpw (no emb {r['bpw_no_emb']:.4f}) from {r['from']} pred {r['pred_kl']:.5f} head {r['head']} emb {r['emb']} "
              f"| uniform {r['vs_uniform']['uniform_kl']:.5f} ch {len(ch)} | K {min(hist):g}-{max(hist):g}", flush=True)
    M.fixed = fixed0
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
