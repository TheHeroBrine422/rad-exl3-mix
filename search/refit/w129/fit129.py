#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fit129.py SPECS.jsonl RESULTS.jsonl DOCSDIR MENU.json OUTDIR -- #169's refit (search/refit/w129/RULES-prior.md):
the calibration swaps as search/v3/fit.py rows, the prior test (β 0 / fit / 1 fitted on U + G, predicting P), the
held-out check H on s129, the ladder flag, the final refit on every row and the fitted per-size slopes b.
Runs on colo (scipy; single-threaded, a few min). Outputs in OUTDIR:
  prior_obs129.jsonl   the rows as the driver's CONF prior_obs (tag, assign, label, kld, se, size, kind)
  fit129.out.json      fit.py's output format for the final model ({report, model, x, draws, obs_tags}); its menu = MENU
  report129.json       every number below; a summary on stdout."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import fit as FT    # noqa: E402
import menu as MN   # noqa: E402
import moves as MV  # noqa: E402

ANCHOR = "w129-u-k2.5"
OFFSET0 = 8255936                      # container - menu bytes (w126, search/v3/README.md)


def load_rows(specs, results, docsdir):
    S = {}; R = {}
    for l in open(specs):
        s = json.loads(l); S[s["tag"]] = s
    for l in open(results):
        r = json.loads(l)
        if r.get("tag") in S and not r.get("fail"): R[(r["tag"], r["slice"])] = r
    D = {}
    for (t, sl) in R:
        p = f"{docsdir}/{t}-{sl}.json"
        d = json.load(open(p)); D[(t, sl)] = np.array([x for pn in sorted(d) for x in d[pn]], float)   # [sum, n] per doc
    return S, R, D


def lse(D, terms):
    """SE of sum_k c_k log(KLD_k) on shared documents (delta method, per-document clusters): terms [(key, c)]"""
    z = 0.0
    for k, c in terms:
        a = D[k][:, 0]; z = z + c * a / a.sum()
    return float(np.sqrt(np.sum((z - np.mean(z)) ** 2) * len(z) / max(len(z) - 1, 1)))


def assign_of(s):
    a = dict(s["mix"]["levels"]); a[MN.HEAD] = "base"; a[MN.EMB] = "base"; return a


def kfloat(name): return float(name.rsplit("-k", 1)[1].split("-")[0])


def main(specs, results, docsdir, menu, outdir):
    os.makedirs(outdir, exist_ok=True)
    S, R, D = load_rows(specs, results, docsdir); M = MN.Menu.load(menu)
    rows = []
    for (t, sl), r in sorted(R.items()):
        s = S[t]; anc = (ANCHOR, sl)
        se = 0.0 if t == ANCHOR or anc not in D else lse(D, [((t, sl), 1), (anc, -1)])
        a = assign_of(s)
        rows.append({"tag": f"{t}@{sl}", "assign": a, "label": sl, "kld": r["kld"], "se": se,
                     "size": round(M.bpw(a) + OFFSET0 * 8 / M.P, 4), "kind": s["w129"]["kind"], "bpw_meas": r["bpw"]})
    by = {o["tag"]: o for o in rows}
    with open(f"{outdir}/prior_obs129.jsonl", "w") as f:
        for o in rows: f.write(json.dumps(o) + "\n")
    rep = {"rows": len(rows), "missing": sorted(f"{t}@{x['name']}" for t, s in S.items() for x in s["slices"] if (t, x["name"]) not in R)}

    # ---- ladder check (U on fit129)
    U = sorted([(kfloat(s["mix"]["levels"]["blk.0.ffn_down.weight"]), t) for t, s in S.items() if s["w129"]["kind"] == "U" and (t, "fit129") in R])
    lad = []
    for j, (k, t) in enumerate(U):
        e = {"K": k, "tag": t, "kld": R[(t, "fit129")]["kld"], "bpw": R[(t, "fit129")]["bpw"], "flags": []}
        if j > 0:
            kp, tp = U[j - 1]
            if e["kld"] >= R[(tp, "fit129")]["kld"]: e["flags"].append(f"not below {tp}")
        if 0 < j < len(U) - 1:
            (k0, t0), (k1, t1) = U[j - 1], U[j + 1]; w1 = (k - k0) / (k1 - k0)
            dev = math.log(e["kld"]) - ((1 - w1) * math.log(R[(t0, "fit129")]["kld"]) + w1 * math.log(R[(t1, "fit129")]["kld"]))
            se = lse(D, [((t, "fit129"), 1), ((t0, "fit129"), -(1 - w1)), ((t1, "fit129"), -w1)])
            e["dev_log"] = dev; e["dev_se"] = se
            if dev > max(3 * se, 0.02): e["flags"].append(f"log KLD {dev:+.3f} above its neighbours' line (SE {se:.3f})")
        lad.append(e)
    rep["ladder"] = lad

    # ---- the prior test: fit on U + G (fit129), predict P; H on s129
    train = [o for o in rows if o["label"] == "fit129" and o["kind"] in ("U", "G")]
    P = [o for o in rows if o["label"] == "fit129" and o["kind"] == "P"]
    H = [o for o in rows if o["label"] == "s129"]
    base = by.get(f"{ANCHOR}@fit129")
    arms = {}
    for arm, beta in (("beta0", "0"), ("beta-fit", "fit"), ("beta1", "1")):
        Mo, x, _, frep = FT.fit_obs(M, train, {"beta": beta})
        tb = Mo.tables(x); pb = Mo.predict(x, base["assign"], tb)
        d = {}
        for o in P:
            d[o["tag"]] = (math.log(Mo.predict(x, o["assign"], tb) / pb) - math.log(o["kld"] / base["kld"]),
                           math.log(Mo.predict(x, o["assign"], tb) / pb), math.log(o["kld"] / base["kld"]))
        fam = {}
        for t, s in S.items():
            if s["w129"]["kind"] != "P": continue
            fam.setdefault(s["w129"]["family"], {})[s["w129"]["arm"]] = f"{t}@fit129"
        order = {}
        for f_, ar in fam.items():
            if not all(k in d for k in ar.values()) or len(ar) < 2: order[f_] = None; continue
            pr = d[ar["hi"]][1] - d[ar["lo"]][1]; me = d[ar["hi"]][2] - d[ar["lo"]][2]
            se = lse(D, [((ar["hi"].split("@")[0], "fit129"), 1), ((ar["lo"].split("@")[0], "fit129"), -1)])
            order[f_] = {"pred_log_hi_lo": pr, "meas_log_hi_lo": me, "meas_se": se, "right": bool((pr > 0) == (me > 0))}
        hh = {}
        for o in H:
            t = o["tag"].split("@")[0]; s = S[t]
            if s["w129"]["kind"] != "G": continue
            bt = f"w129-u-k{s['w129']['base']:g}@s129"
            if bt not in by: continue
            b_ = by[bt]; lp = math.log(Mo.predict(x, o["assign"], tb) / Mo.predict(x, b_["assign"], tb))
            parts = {}
            for pn in ("x", "c"):
                ra, rb = R[(t, "s129")]["parts"].get(pn), R[(bt.split("@")[0], "s129")]["parts"].get(pn)
                if ra and rb: parts[pn] = math.log(ra["kld"] / rb["kld"])
            hh[o["tag"]] = {"pred": lp, "meas": math.log(o["kld"] / b_["kld"]), "meas_parts": parts}
        u = Mo.unpack(x)
        arms[arm] = {"beta": float(u["beta"]), "q": float(u["q"]), "train": frep,
                     "P_rmse": float(np.sqrt(np.mean([v[0] ** 2 for v in d.values()]))) if d else None, "P_n": len(d),
                     "P_rows": {k: {"err": v[0], "pred": v[1], "meas": v[2]} for k, v in d.items()},
                     "order": order, "order_right": sum(1 for v in order.values() if v and v["right"]),
                     "H_rmse": float(np.sqrt(np.mean([(v["pred"] - v["meas"]) ** 2 for v in hh.values()]))) if hh else None,
                     "H_rmse_x": float(np.sqrt(np.mean([(v["pred"] - v["meas_parts"]["x"]) ** 2 for v in hh.values() if "x" in v["meas_parts"]]))) if hh else None,
                     "H_rmse_c": float(np.sqrt(np.mean([(v["pred"] - v["meas_parts"]["c"]) ** 2 for v in hh.values() if "c" in v["meas_parts"]]))) if hh else None,
                     "H_rows": hh}
    rep["arms"] = arms
    a0, af = arms["beta0"], arms["beta-fit"]
    ok = af["P_rmse"] is not None and a0["P_rmse"] is not None and af["P_rmse"] < a0["P_rmse"] and af["order_right"] >= 6
    rep["verdict"] = "PASS" if ok else "FAIL"
    beta = "fit" if ok else "0"

    # ---- final refit on every row, slopes b per size
    Mo, x, draws, frep = FT.fit_obs(M, rows, {"beta": beta}, ndraw=16, seed=169)
    out = {"report": frep, "model": Mo.dump(x), "x": x.tolist(), "draws": [v.tolist() for v in draws],
           "obs_tags": [o["tag"] for o in rows], "w129": {"menu": menu, "beta_mode": beta, "verdict": rep["verdict"]}}
    json.dump(out, open(f"{outdir}/fit129.out.json.part", "w")); os.replace(f"{outdir}/fit129.out.json.part", f"{outdir}/fit129.out.json")
    pred = MV.Pred(Mo, x); lv = M.levels(); P_ = M.P; slopes = {}
    off = OFFSET0 * 8 / P_
    for size in [1.5 + 0.25 * i for i in range(23)]:
        pts = []
        for s2 in (size - 0.1, size + 0.1):
            hi = (s2 - off) * P_ / 8; lo = (s2 - off - 0.02) * P_ / 8
            under = [n for n in lv if M.bpw(M.uniform(n, "base", "base")) + off <= s2]
            if not under: continue
            m = MV.mckp(pred, M.uniform(under[-1], "base", "base"), hi, lo)
            if m is not None: pts.append((M.bpw(m) + off, math.log(pred.kl(m))))
        if len(pts) == 2 and pts[1][0] - pts[0][0] > 1e-4:
            slopes[f"{size:g}"] = {"b": -(pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0]), "kl": math.exp(0.5 * (pts[0][1] + pts[1][1]))}
    rep["final"] = {"beta_mode": beta, "beta": float(Mo.unpack(x)["beta"]), "report": frep, "slopes": slopes,
                    "kappa": out["model"]["kappa"][::8], "q": out["model"]["q"], "phi3": out["model"]["phi3"], "slices": out["model"]["slices"]}
    json.dump(rep, open(f"{outdir}/report129.json", "w"), indent=1, default=float)

    print(f"rows {len(rows)} (missing {rep['missing']})")
    for e in lad: print(f"  U K{e['K']:<5g} kld {e['kld']:.5f} bpw {e['bpw']:.3f} dev {e.get('dev_log', float('nan')):+.3f} ± {e.get('dev_se', float('nan')):.3f} {'FLAG ' + '; '.join(e['flags']) if e['flags'] else ''}")
    for k, v in arms.items():
        print(f"  [{k}] beta {v['beta']:.3f} q {v['q']:.3f} train rmse {v['train']['rmse_log']:.4f} | P rmse {v['P_rmse']:.4f} (n {v['P_n']}) "
              f"order {v['order_right']}/8 | H rmse {v['H_rmse']} (x {v['H_rmse_x']}, c {v['H_rmse_c']})")
    for f_, v in arms["beta-fit"]["order"].items():
        if v: print(f"    {f_:24s} meas log(hi/lo) {v['meas_log_hi_lo']:+.4f} ± {v['meas_se']:.4f}  pred(β fit) {v['pred_log_hi_lo']:+.4f}  "
                    f"pred(β0) {arms['beta0']['order'][f_]['pred_log_hi_lo']:+.4f}  {'ok' if v['right'] else 'WRONG'}")
    print(f"VERDICT {rep['verdict']} -> final refit beta {beta}: {frep}")
    print("slopes b: " + ", ".join(f"{k} {v['b']:.2f}" for k, v in slopes.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
