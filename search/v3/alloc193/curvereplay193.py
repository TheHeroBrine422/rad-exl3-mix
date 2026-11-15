# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""curvereplay193.py SNAP CURV.json CURVEFIT.json CANDS.jsonl OUT.json [--step 300] -- RULES-193 §6: time-ordered
comparison of the §1 curve model with fit.py (the live pre-submit pred_logr, fitted on results up to ~submit time).

At refit points N = step, 2 step, ...: the curve model (linear solver, kappa / kappa_x / sigma fixed at CURVEFIT's) is
fitted on the fit rows whose eval had a result <= N (prior rows always), with its prior = the incumbents' pooled curve at
N (local-linear, equal weights; no s186e / s190a re-selection information from the future). It predicts the stage-1
candidates (rank193 cands table) whose results are in (N, N + step]: log r = kappa [q(cand) - q(ref)] at the nominal size.
Metrics: RULES-187 band_metrics (rho per band, S) + paired bootstrap vs pred_live on the same rows.
Caveats (reported): the curvature table and the hyperparameters come from the full-data refit / CV."""
import argparse, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rerank187")); sys.path.insert(0, os.path.join(HERE, ".."))
import common193 as C  # noqa: E402
import cons193 as CS  # noqa: E402
import curve193 as CU  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("curv"); ap.add_argument("cfit"); ap.add_argument("cands")
    ap.add_argument("out"); ap.add_argument("--step", type=int, default=300)
    a = ap.parse_args()
    import metrics as MT
    M = C.load_menu(a.snap); km = C.KM(M); curv = json.load(open(a.curv)); F = json.load(open(a.cfit))
    kb, kx, sg = np.array(F["kappa"]), F["kappa_x"], F["sigma"]
    st = C.load_state(a.snap); nres = {}; n = 0; inc_t = []
    for e in st:
        if e.get("ev") == "result": n += 1; nres.setdefault(e["tag"], n)
        if e.get("ev") == "incumbent": inc_t.append((n, float(e["size"]), e["tag"]))
    D = CU.Data(a.snap, curv, km)
    ridx = np.array([nres.get(t, 0) for t in D.tags])                          # prior rows (no result event) -> 0
    cands = [json.loads(l) for l in open(a.cands)]
    cs = sorted(float(k) for k in curv["curv"])
    cvec_s = {}
    def cvec(s):
        if s not in cvec_s: cvec_s[s] = np.array([np.interp(s, cs, [curv["curv"][f"{x:g}"][w][1] for x in cs]) for w in km.W])
        return cvec_s[s]
    Kc = {}
    def kv(tag):
        if tag not in Kc: asg = C.assign(a.snap, tag); Kc[tag] = np.array([km.Kof[w][asg[w]] for w in km.W])
        return Kc[tag]
    pred = {}; maxN = max(c["result_idx"] for c in cands)
    ss = np.array(C.SIZES); Bm = CU.basis(ss); Ap = Bm.T @ Bm + np.diag([0, 0, 1e-3])
    for N in range(a.step, maxN + 1, a.step):
        te = [c for c in cands if N < c["result_idx"] <= N + a.step and c["size"] in C.BAND]
        if not te: continue
        cur = {}
        for n_, s, t in inc_t:
            if n_ <= N: cur[s] = t
        cv = CS.Curve(km, [(s, 1.0, C.assign(a.snap, t), t) for s, t in cur.items()])
        p0 = np.array([np.linalg.solve(Ap, Bm.T @ (np.array([cv.target(w, s) for s in ss]) - ss)) for w in km.W])
        idx = np.where(ridx <= N)[0]
        th, off = CU.lin_solve(D, kb, kx, p0, sg, idx)
        for c in te:
            s = c["size"]; Ct = s + CU.basis(np.array([s]))[0] @ th.T; cc = cvec(s)
            q = lambda K: float(np.sum(cc * (K - Ct) ** 2))
            pred[c["tag"]] = float(kb[CU.band(s)] * (q(kv(c["tag"])) - q(kv(c["ref"]))))
        print(f"N {N}: train rows {len(idx)}, predicted {len(te)}", flush=True)
    rows = {"a": [], "curve": []}
    for c in cands:
        if c["tag"] not in pred or c.get("pred_live") is None: continue
        for arm, p in (("a", c["pred_live"]), ("curve", pred[c["tag"]])):
            rows[arm].append({"tag": c["tag"], "band": c["band"], "stopped": c["stopped"], "audit": c["audit"], "meas": c["logr"],
                              "pred": p, "chunk": a.step * ((c["result_idx"] - 1) // a.step)})
    # ensemble in prediction space: per band, the mean of the two arms' within-band ranks
    rows["ens"] = []
    for b in ("b1", "b2", "b3", "b4"):
        ia = [i for i, r in enumerate(rows["a"]) if r["band"] == b]
        if not ia: continue
        from scipy.stats import rankdata
        ra = rankdata([rows["a"][i]["pred"] for i in ia]); rc = rankdata([rows["curve"][i]["pred"] for i in ia])
        for k, i in enumerate(ia):
            r = dict(rows["a"][i]); r["pred"] = float((ra[k] + rc[k]) / 2 / len(ia) - 0.5); rows["ens"].append(r)
    tab = {arm: MT.band_metrics(rows[arm], arm) for arm in rows}
    boot = MT.boot_draws(rows, "curve", "a"); boot_ens = MT.boot_draws(rows, "ens", "a")
    res = {"table": tab, "boot": boot, "boot_ens": boot_ens, "n": len(rows["a"]), "step": a.step}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    for arm in rows:
        t = tab[arm]
        print(f"{arm}: S {t['S']:.3f} rho b1 {t['b1']['rho']:.3f} b2 {t['b2']['rho']:.3f} b3 {t['b3']['rho']:.3f} "
              f"(whole {t['b1']['n_whole']}/{t['b2']['n_whole']}/{t['b3']['n_whole']}) hit b1/b2/b3 {t['b1']['hit']:.2f}/{t['b2']['hit']:.2f}/{t['b3']['hit']:.2f}")
    print("dS curve - a:", boot); print("dS ens - a:", boot_ens)


if __name__ == "__main__":
    main()
