# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fitopt193.py SNAP RDIR OUTDIR --alloc tr|rob|recal|ens [--sizes 2.5,3,4,5] [--anom ANOM.json] [--curve CURVEFIT.json
--curv CURV.json] [--commit C] -- RULES-193 amendment 2: cautious optimizers on my fit.py refit (RDIR/refit-<band>.*,
RDIR/draw-<band>.out.json for the posterior draws).

All start at the size's current incumbent and keep its embedding option. Per weight the allowed options are its own
family's within +-R K of the incumbent's (R = 0.25). Objective per weight and option = a linear value v_tj (change of
predicted log KL vs the incumbent's option); the multiple-choice knapsack picks argmin v + lam * bytes (bisection on
lam to the window top), then fills the window with the best value per byte; fit.py's part is re-linearised 3 times
(as moves.mckp).
  tr    : v = fit.py's linearised d log KL
  recal : v = s_band * fit.py + beta_g * n_t (K_j - K_t) / sum n   (anom193 version ii, s_band = the band's slope of
          measured on fit.py-predicted log r over the same paired rows)
  ens   : v = 1/2 fit.py + 1/2 curve model (kappa_band c_t ((K_j - C_t)^2 - (K_t - C_t)^2))
  rob   : candidates = tr optima at R 0.125 / 0.25 / 0.5 under the mean and each posterior draw; pick the best 80th
          percentile of predicted log r vs the incumbent over the draws."""
import argparse, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import common193 as C  # noqa: E402
import curv193 as CV  # noqa: E402
import moves as MV  # noqa: E402
import menu as MN  # noqa: E402


def load(M, rdir, g, draws=False):
    d = json.load(open(f"{rdir}/refit-{g}.in.json")); import fit as FT
    Mo = FT.build(M, d["obs"], d.get("opts", {}))
    o = json.load(open(f"{rdir}/refit-{g}.out.json")); P = MV.Pred(Mo, np.array(o["x"])); Ds = []
    if draws and os.path.exists(f"{rdir}/draw-{g}.out.json"):
        Ds = [MV.Pred(Mo, np.array(v)) for v in json.load(open(f"{rdir}/draw-{g}.out.json"))["draws"]]
    return P, Ds


def optimize(pred, ia, M, s0, R, extra=None, w_fit=1.0, iters=3):
    """pred: moves.Pred; extra(i, w, Kopt array, Kcur) -> additive value per option (or None)"""
    lo, hi = C.window(M, s0); D = pred.D; W = D.W
    fam = {w: C.KM.fam(ia[w]) for w in W}
    Kopt = [np.array([o["K"] for o in M.T[M.WI[w]]["opts"]]) for w in W]
    Kinc = np.array([M.opt(w, ia[w])["K"] for w in W])
    allow = [np.array([(C.KM.fam(o["o"]) == fam[w]) and abs(o["K"] - Kinc[i]) <= R + 1e-9 for o in M.T[M.WI[w]]["opts"]])
             for i, w in enumerate(W)]
    cur = dict(ia); fixed = M.bytes(ia) - sum(pred.B[i][j] for i, j in enumerate(pred.idx(ia)))
    for _ in range(iters):
        J0, dl = pred.lin(cur)
        Kc = np.array([Kopt[i][J0[i]] for i in range(len(W))])
        # values relative to the incumbent: fit.py part linearised at cur, extras exact vs the incumbent
        V = []
        for i, w in enumerate(W):
            v = w_fit * dl[i].copy()
            if extra is not None: v = v + extra(i, w, Kopt[i], Kinc[i]) - extra(i, w, np.array([Kc[i]]), Kinc[i])[0]
            v[~allow[i]] = np.inf; V.append(v)
        def pick(lam):
            J = [int(np.argmin(V[i] + lam * pred.B[i])) for i in range(len(W))]
            return J, fixed + sum(pred.B[i][j] for i, j in enumerate(J))
        J, b = pick(0.0)
        if b > hi:
            l0, l1 = 0.0, 1e-15
            while pick(l1)[1] > hi: l0 = l1; l1 *= 4
            for _ in range(60):
                m = (l0 + l1) / 2
                if pick(m)[1] > hi: l0 = m
                else: l1 = m
            J, b = pick(l1)
        # fill: best value per byte that fits (only improving moves)
        while True:
            best = None
            for i in range(len(W)):
                j = J[i]
                for jj in range(j + 1, len(V[i])):
                    db = pred.B[i][jj] - pred.B[i][j]
                    if db <= 0 or b + db > hi or not np.isfinite(V[i][jj]): continue
                    gain = V[i][j] - V[i][jj]
                    if gain <= 0: continue
                    if best is None or gain / db > best[0]: best = (gain / db, i, jj, db)
            if best is None: break
            J[best[1]] = best[2]; b += best[3]
        cur = pred.assign(J, cur)
    if M.bytes(cur) < lo: return None
    return cur


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("rdir"); ap.add_argument("out")
    ap.add_argument("--alloc", required=True); ap.add_argument("--sizes", default="2.5,3,4,5"); ap.add_argument("--R", type=float, default=0.25)
    ap.add_argument("--anom"); ap.add_argument("--curve"); ap.add_argument("--curv"); ap.add_argument("--commit", default="?")
    a = ap.parse_args()
    M = C.load_menu(a.snap); km = C.KM(M); st = C.load_state(a.snap); inc = {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
    os.makedirs(a.out, exist_ok=True); cache = {}
    HYP = {"tr": "fit.py optimum in a +-0.25 K trust region beats the incumbent (optimizer's curse avoided)",
           "rob": "fit.py's pessimistic (80th pct over posterior draws) optimum beats the incumbent",
           "recal": "fit.py recalibrated (band slope + group residuals from paired outcomes), +-0.25 trust region, beats the incumbent",
           "ens": "mean of fit.py and the curve model, +-0.25 trust region, beats the incumbent"}
    for s0 in [float(x) for x in a.sizes.split(",")]:
        g = CV.band(s0)
        if g not in cache: cache[g] = load(M, a.rdir, g, draws=(a.alloc == "rob"))
        pred, draws = cache[g]; ia = C.assign(a.snap, inc[s0]); W = pred.D.W
        extra = None; wf = 1.0
        if a.alloc == "recal":
            A = json.load(open(a.anom)); co = A["ii"]["groups"]; sl = A["slope"][g]
            n = {}
            for t in M.T:
                o = sorted(t["opts"], key=lambda x: x["bytes"]); n[t["w"]] = 8 * (o[-1]["bytes"] - o[0]["bytes"]) / max(o[-1]["K"] - o[0]["K"], 1e-9)
            tot = sum(n.values()); wf = sl
            zmin = float(os.environ.get("RECAL_ZMIN", "0"))      # amendment 3: only |z| >= 2 coefficients
            beta = {w: (co[C.gname(w)][g]["beta"] if abs(co[C.gname(w)][g]["beta"]) >= zmin * co[C.gname(w)][g]["se"] else 0.0) for w in W}
            extra = lambda i, w, K, K0: beta[w] * n[w] * (K - K0) / tot
        elif a.alloc == "ens":
            F = json.load(open(a.curve)); cp = np.array(F["cp"]); cu = json.load(open(a.curv)); kb = F["kappa"][[0, 1, 2][0 if s0 < 2.75 else (1 if s0 <= 4.25 else 2)]]
            cs = sorted(float(k) for k in cu["curv"]); FW = F["W"]
            import curve193 as CU
            Ct = dict(zip(FW, s0 + CU.basis(np.array([s0]))[0] @ cp.T))
            cvec = {w: np.interp(s0, cs, [cu["curv"][f"{s:g}"][w][1] for s in cs]) for w in FW}
            wf = 0.5
            extra = lambda i, w, K, K0: 0.5 * kb * cvec[w] * ((K - Ct[w]) ** 2 - (K0 - Ct[w]) ** 2)
        if a.alloc == "rob":
            cands = []
            for R in (0.125, 0.25, 0.5):
                for p in [pred] + draws:
                    c = optimize(p, ia, M, s0, R)
                    if c is not None: cands.append((R, c))
            kli = [d.kl(ia) for d in draws]
            scored = []
            for R, c in cands:
                lr = np.array([math.log(d.kl(c) / k0) for d, k0 in zip(draws, kli)])
                scored.append((float(np.percentile(lr, 80)), R, c))
            scored.sort(key=lambda z: z[0]); q80, R, asg = scored[0]
            info = f"q80 {q80:+.4f} (R {R}, of {len(scored)})"
        else:
            asg = optimize(pred, ia, M, s0, a.R, extra, wf); info = ""
        if asg is None: print(f"{s0:g}: no feasible allocation"); continue
        pm = math.log(pred.kl(asg) / pred.kl(ia))
        nd = sum(asg[w] != ia[w] for w in km.W); dk = np.mean([abs(km.Kof[w][asg[w]] - km.Kof[w][ia[w]]) for w in km.W])
        d = {"size": s0, "mode": "decide", "assign": asg, "note": f"#193 {a.alloc} | {HYP[a.alloc]} | code {a.commit}"}
        why = C.check_file(M, d)
        print(f"{a.alloc} {s0:g}: bpw {M.bpw(asg):.4f} vs {inc[s0]}: {nd} changed, mean|dK| {dk:.3f}, fit.py log r {pm:+.4f} {info} {why or 'OK'}", flush=True)
        if why is None: json.dump(d, open(f"{a.out}/{a.alloc}-{s0:g}.json", "w"))


if __name__ == "__main__":
    main()
