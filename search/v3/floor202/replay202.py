# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""replay202.py -- RULES-floor202 §2: time-ordered replay of fitfloor (fit202.py, opts floor) vs fit.py on the b3 rows.

  replay202.py cands DRV OUT.json [STRIDE]   (VM) b3 candidates (sizes 4.5-7.0; stage-1 decisions / tentatives with a log r,
                                             as rank193.py cands, every search slice), their and their references' assignments,
                                             the incumbents per b3 fit, and the replay's fit points: every STRIDE-th (3) b3
                                             fit_done of the live run. A candidate is predicted by the last fit point whose
                                             n_results is below its result index (the fit saw no row of it: checked by tag).
  replay202.py pred CANDS.json FITDIR K OUTDIR   (colo) fit point K: arm a = the live fit.py output (FITDIR/<fit>.out.json, the
                                             driver's own fit on <fit>.in.json); arm f = fit202 floor on the same in.json,
                                             started from arm a's vector (+ log F at its prior mean). Predictions
                                             log(KL(cand) / KL(ref)) on each model, F +- SE, and the driver's slope b
                                             (driver.slope: knapsack optimum at size -/+ 0.1 from the incumbent) per size.
  replay202.py table CANDS.json OUTDIR OUT   metrics (rerank187/metrics.py's Spearman on whole-slice rows; pair accuracy as
                                             rank193.py: same size x slice x reference) per size; S(sizes) = mean of the sizes'
                                             rho; paired bootstrap of dS = S(f) - S(a), rows resampled within size, 2000 draws,
                                             seed 202, 80 % CI. Gate (RULES §2): dS CI wholly above 0 for sizes 6.0-7.0."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
BANDS = {"b1": [1.5, 1.75, 2.0, 2.5, 2.75], "b2": [3.0, 3.25, 3.5, 3.75, 4.0], "b3": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]}   # 1.25 is stopped
SETS = {"b3": {"6.0-7.0 (gate, RULES §2)": [6.0, 6.5, 7.0], "5.5-7.0": [5.5, 6.0, 6.5, 7.0]}}
GEN = {"solve", "partial", "pair", "block", "thompson", "explore", "cross", "manual"}
TOL, DB = 0.02, 0.1


def cands(drv, out, stride=3, band="b3"):
    SIZES = BANDS[band]
    st = [json.loads(l) for l in open(f"{drv}/state.jsonl")]
    sub = {e["tag"]: e for e in st if e.get("ev") == "submit"}
    nres, n, res, fits, inc, inc_at = {}, 0, {}, [], {}, []
    for e in st:
        if e.get("ev") == "result": n += 1; nres.setdefault(e["tag"], n); res[e["tag"]] = e["r"]
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
        if e.get("ev") == "fit_done" and e.get("group") == band:
            fits.append({"name": os.path.basename(e["out"])[:-len(".out.json")], "n_results": e["n_results"], "t": e.get("t"),
                         "inc": {str(s): inc[s] for s in SIZES if s in inc}})
    sel = fits[::stride]
    rows, need = [], set()
    for e in st:
        if e.get("ev") not in ("decision", "tentative") or e.get("stage") == 2 or e.get("logr") is None: continue
        s = sub.get(e["tag"])
        if s is None or s["kind"] not in GEN or float(s["size"]) not in SIZES or e["tag"] not in nres: continue
        ref = e.get("vs") or s.get("ref")
        if not ref or not os.path.exists(f"{drv}/assign/{ref}.json"): continue
        k = max([i for i, f in enumerate(sel) if f["n_results"] < nres[e["tag"]]], default=None)
        if k is None: continue
        rows.append({"tag": e["tag"], "size": float(s["size"]), "kind": s["kind"], "slice": s["slice"], "ref": ref, "result_idx": nres[e["tag"]],
                     "stopped": bool(res[e["tag"]].get("stopped")), "audit": bool(s.get("audit")), "decision": e.get("decision") or "tentative",
                     "logr": e["logr"], "pred_live": s.get("pred_logr"), "fit": k})
        need |= {e["tag"], ref}
    for f in sel: need |= set(f["inc"].values())
    A = {t: json.load(open(f"{drv}/assign/{t}.json")) for t in sorted(need)}
    json.dump({"band": band, "sizes": SIZES, "fits": sel, "all_fits": len(fits), "stride": stride, "cands": rows, "assign": A}, open(out, "w"))
    by = {}
    for r in rows: by.setdefault(r["size"], [0, 0]); by[r["size"]][0] += 1; by[r["size"]][1] += not r["stopped"]
    print(len(rows), "candidates,", len(sel), "of", len(fits), "fit points; per size (all, whole):", {s: by[s] for s in sorted(by)})


def slope(M, pred, a0, size):
    """driver.slope on a prediction (same windows and mckp)"""
    import moves as MV
    minb = M.fixed + sum(t["opts"][0]["bytes"] for t in M.T); maxb = M.fixed + sum(t["opts"][-1]["bytes"] for t in M.T); pts = []
    for s2 in (size - DB, size, size + DB):
        hi = s2 * M.P / 8; lo = (s2 - TOL) * M.P / 8
        if minb > hi or maxb < lo: continue
        m = MV.mckp(pred, a0, hi, lo)
        if m is not None: pts.append((M.bpw(m), math.log(pred.kl(m))))
    if len(pts) < 2 or pts[-1][0] - pts[0][0] < 1e-4: return None
    b = -(pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
    return b if math.isfinite(b) else None


def pred(cpath, fitdir, k, outdir, term="floor"):
    import fit as FT, fit202 as F2, menu as MN, moves as MV
    C = json.load(open(cpath)); f = C["fits"][k]; name = f["name"]
    outp = os.path.join(outdir, f"{k:03d}.json")
    if os.path.exists(outp): return
    d = json.load(open(f"{fitdir}/{name}.in.json")); live = json.load(open(f"{fitdir}/{name}.out.json"))
    M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
    opts = d.get("opts") or {}
    Ma = FT.build(M, d["obs"], opts); xa = np.array(live["x"]); Ma.layout(); assert len(xa) == Ma.N
    fo = dict(opts, **{term: True})
    Mf, xf, _, rep = F2.fit_obs(M, d["obs"], fo, list(xa))
    P = {"a": MV.Pred(Ma, xa), "f": MV.Pred(Mf, xf)}
    seen = {t.split("@")[0] for t in live["obs_tags"]}
    te = [c for c in C["cands"] if c["fit"] == k]; out = {"a": {}, "f": {}}; leak = 0
    for c in te:
        if c["tag"] in seen: leak += 1; continue
        for arm in P: out[arm][c["tag"]] = math.log(P[arm].kl(C["assign"][c["tag"]]) / P[arm].kl(C["assign"][c["ref"]]))
    b = {arm: {s: slope(M, P[arm], C["assign"][t], float(s)) for s, t in f["inc"].items()} for arm in P}
    res = {"fit": name, "k": k, "n_results": f["n_results"], "rows": len(d["obs"]), "preds": out, "leak": leak, "b": b,
           "term": term, "par": rep["F"] if term == "floor" else rep["S"], "log_se": rep["logF_se"] if term == "floor" else rep["logS_se"],
           "report_f": rep, "report_a": live["report"], "xf": xf.tolist()}
    json.dump(res, open(outp + ".part", "w")); os.replace(outp + ".part", outp)
    print(name, "cands", len(te), "leak", leak, term, "%.3g se(log) %.3f" % (res["par"], res["log_se"]), "cost a %.1f f %.1f nfev %d" % (live["report"]["cost"], rep["cost"], rep["nfev"]), "b", b)


def _sp(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0: return float("nan")
    return float(spearmanr(x, y).correlation)


def table(cpath, outdir, out):
    C = json.load(open(cpath)); R = {}; band = C.get("band", "b3"); SIZES = C.get("sizes", BANDS["b3"])
    files = sorted(f for f in os.listdir(outdir) if f.endswith(".json") and f[:3].isdigit())
    runs = [json.load(open(os.path.join(outdir, f))) for f in files]
    term = runs[0].get("term", "floor")
    for r in runs:
        for t, v in r["preds"]["a"].items(): R[t] = (v, r["preds"]["f"][t])
    cs = [c for c in C["cands"] if c["tag"] in R and all(math.isfinite(v) for v in R[c["tag"]])]
    W = [c for c in cs if not c["stopped"]]
    arr = {s: np.array([(R[c["tag"]][0], R[c["tag"]][1], c["logr"]) for c in W if c["size"] == s]).reshape(-1, 3) for s in SIZES}

    def S(get, sizes): return float(np.nanmean([get(s) for s in sizes]))
    per = {}
    for s in SIZES:
        A = arr[s]; allr = [c for c in cs if c["size"] == s]
        m = {"n_whole": len(A), "n_all": len(allr)}
        for j, arm in enumerate("af"):
            x = A[:, j]; y = A[:, 2]
            pn = [c for c in allr if R[c["tag"]][j] < 0]
            m[arm] = {"rho": _sp(x, y), "slope": float(np.polyfit(x, y, 1)[0]) if len(A) >= 3 and np.ptp(x) > 0 else float("nan"),
                      "overstatement": float(np.mean(x[x < 0]) / np.mean(y[x < 0])) if (x < 0).sum() >= 2 else float("nan"),
                      "hit": sum(1 for c in pn if not c["stopped"] and c["logr"] < 0) / len(pn) if pn else float("nan"), "n_pred_better": len(pn)}
        per[s] = m
    # pair accuracy: whole rows of the same size x slice x reference
    byg = {}
    for c in W: byg.setdefault((c["size"], c["slice"], c["ref"]), []).append(c)
    pacc = {s: {"a": [0, 0], "f": [0, 0]} for s in SIZES}
    for (s, _, _), L in byg.items():
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                dm = L[i]["logr"] - L[j]["logr"]
                for q, arm in enumerate("af"):
                    dp = R[L[i]["tag"]][q] - R[L[j]["tag"]][q]
                    if dm == 0 or dp == 0: continue
                    pacc[s][arm][0] += (dm > 0) == (dp > 0); pacc[s][arm][1] += 1
    sets = dict(SETS.get(band, {}), **{f"{SIZES[0]:g}-{SIZES[-1]:g} (band)": SIZES}, **{f"{s:g}": [s] for s in SIZES})
    rng = np.random.default_rng(202); NB = 2000; boot = {k: [] for k in sets}
    for _ in range(NB):
        rho = {}
        for s in SIZES:
            A = arr[s]
            if len(A) < 3: rho[s] = (float("nan"), float("nan")); continue
            Q = A[rng.integers(0, len(A), len(A))]; rho[s] = (_sp(Q[:, 0], Q[:, 2]), _sp(Q[:, 1], Q[:, 2]))
        for k, ss in sets.items(): boot[k].append(S(lambda s: rho[s][1], ss) - S(lambda s: rho[s][0], ss))
    summ = {}
    for k, ss in sets.items():
        Sa, Sf = S(lambda s: per[s]["a"]["rho"], ss), S(lambda s: per[s]["f"]["rho"], ss); b = np.array(boot[k])
        pa = {arm: sum(pacc[s][arm][0] for s in ss) / max(sum(pacc[s][arm][1] for s in ss), 1) for arm in "af"}
        ci = [float(np.nanpercentile(b, 10)), float(np.nanpercentile(b, 90))] if np.isfinite(b).any() else [float("nan")] * 2
        summ[k] = {"sizes": ss, "S_a": Sa, "S_f": Sf, "dS": Sf - Sa, "dS_ci80": ci, "pass": bool(ci[0] > 0), "live": bool(Sf - Sa >= 0),
                   "pair_acc_a": pa["a"], "pair_acc_f": pa["f"], "pairs": sum(pacc[s]["a"][1] for s in ss), "n_whole": sum(per[s]["n_whole"] for s in ss)}
    Fs = [(r["k"], r["fit"], r["rows"], r.get("par", r.get("F")), r.get("log_se", r.get("logF_se")), r["b"], r["report_a"]["cost"], r["report_f"]["cost"]) for r in runs]
    res = {"band": band, "term": term, "summary": summ, "per_size": {str(s): per[s] for s in SIZES},
           "fits": [{"k": k, "fit": n, "rows": r, "par": F, "log_se": se, "b": b, "cost_a": ca, "cost_f": cf} for k, n, r, F, se, b, ca, cf in Fs],
           "fit_points_done": len(runs), "fit_points": len(C["fits"]), "leaks": sum(r["leak"] for r in runs)}
    json.dump(res, open(out + ".json", "w"), indent=1)
    pn = "F" if term == "floor" else "S"
    L = [f"# #202 replay: fit{term if term == 'floor' else 'sat'} (f) vs fit.py (a), {band} rows", "",
         f"Fit points {len(runs)} of {len(C['fits'])} (every {C['stride']}rd of the live run's {C['all_fits']} {band} fits); candidates with predictions {len(cs)} ({len(W)} whole-slice); leaks {res['leaks']}.",
         "S = mean over the sizes of Spearman(pred, measured log r) on whole-slice rows; dS = S(f) - S(a), paired bootstrap within size (2000, seed 202).",
         "pass = 80 % CI wholly above 0 (RULES §2); live = dS >= 0 (amendment 2.3).", "",
         "| sizes | S a | S f | dS | dS 80 % CI | pass | live | pair acc a | pair acc f | pairs | whole rows |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in summ.items():
        L.append(f"| {k} | {v['S_a']:.3f} | {v['S_f']:.3f} | {v['dS']:+.4f} | [{v['dS_ci80'][0]:+.4f}, {v['dS_ci80'][1]:+.4f}] | {'pass' if v['pass'] else 'fail'} | {'yes' if v['live'] else 'no'} | "
                 f"{v['pair_acc_a']:.3f} | {v['pair_acc_f']:.3f} | {v['pairs']} | {v['n_whole']} |")
    L += ["", "| size | whole / all | slope a | slope f | overst. a | overst. f | hit a | hit f |", "|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        m = per[s]
        L.append(f"| {s:g} | {m['n_whole']} / {m['n_all']} | {m['a']['slope']:.2f} | {m['f']['slope']:.2f} | "
                 f"{m['a']['overstatement']:.1f} | {m['f']['overstatement']:.1f} | {m['a']['hit']:.2f} | {m['f']['hit']:.2f} |")
    ks = [f"{s:g}" if s != int(s) else f"{s:.1f}" for s in SIZES]
    L += ["", f"| fit point | rows | {pn} | SE(log {pn}) | cost a | cost f | b a ({' / '.join(f'{s:g}' for s in SIZES)}) | b f |", "|---|---|---|---|---|---|---|---|"]
    fb = lambda b: " / ".join("-" if b.get(s) is None else f"{b[s]:.2f}" for s in ks)
    for k, n, r, F, se, b, ca, cf in Fs: L.append(f"| {n} | {r} | {F:.3g} | {se:.3f} | {ca:.1f} | {cf:.1f} | {fb(b['a'])} | {fb(b['f'])} |")
    open(out + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L[:22]))


def shares(inp, fitp, statep, out, live_fit=None):
    """per size at its incumbent, on the fit's calib scale: P, F / (F + P), P / S, the model's KL; the driver's slope b on this fit
    (and on the live fit.py output of the same input, if given)"""
    import fit as FT, fit202 as F2, menu as MN, moves as MV
    d = json.load(open(inp)); ft = json.load(open(fitp)); St = json.load(open(statep))
    M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
    Mo = F2.build(M, d["obs"], ft["opts"]); Mo.layout(); x = np.array(ft["x"]); assert len(x) == Mo.N
    pr = MV.Pred(Mo, x); tb = pr.tb; pa = None
    if live_fit and os.path.exists(live_fit):
        Ma = FT.build(M, d["obs"], d.get("opts") or {}); Ma.layout(); xa = np.array(json.load(open(live_fit))["x"])
        if len(xa) == Ma.N: pa = MV.Pred(Ma, xa)
    res = {"fit": os.path.basename(fitp), "F": tb.get("F"), "S": tb.get("S"), "c0": tb["c0"], "report": ft["report"], "sizes": {}}
    for s, v in sorted(St["inc"].items(), key=lambda kv: float(kv[0])):
        a = v["assign"]; P = FT.SModel.kl_from_X(Mo, tb, pr.X(pr.idx(a)), Mo.D.emb_label(a)); F = tb.get("F") or 0.0; S_ = tb.get("S")
        res["sizes"][s] = {"inc": v["tag"], "P": P, "kl": pr.kl(a), "floor_share": F / (F + (P / (1 + P / S_) if S_ else P)), "P_over_S": P / S_ if S_ else None,
                           "b_f": slope(M, pr, a, float(s)), "b_a": slope(M, pa, a, float(s)) if pa else None, "kl_a": pa.kl(a) if pa else None}
        print(s, {k: (round(x_, 6) if isinstance(x_, float) else x_) for k, x_ in res["sizes"][s].items()})
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "cands": cands(a[1], a[2], int(a[3]) if len(a) > 3 else 3, a[4] if len(a) > 4 else "b3")
    elif a[0] == "pred": pred(a[1], a[2], int(a[3]), a[4], a[5] if len(a) > 5 else "floor")
    elif a[0] == "shares": shares(*a[1:])
    elif a[0] == "table": table(a[1], a[2], a[3])
