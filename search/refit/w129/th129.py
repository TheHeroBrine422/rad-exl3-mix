#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""th129.py SPECS SPECS_TH RESULTS DOCSDIR MENU_H OBS.jsonl OUTDIR -- #169 trunk x head anchors (lead note 11-13 1:00 AM):
uniform K3 / K4 trunks x R16 heads u3 / u4 / u5 (bf16-spliced, priced at the quantized bytes) on fit129. On the R16-head
menu (menu129h): (1) the 84 calibration rows with V2 3.00's head priced as plan P (P = u4, and u2.75, the plan the
measured V2 - u3 gap points to) predict the 6 (held out); (2) the refit on 84 + 6 rows (head = u2.75 for the V2 rows):
in-sample residuals, the trunk x head interaction (log KL(K3, h) - log KL(K4, h) per head, measured vs predicted) and
the driver files OUTDIR/fit129h2.out.json + .obs.jsonl (+ slopes)."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import fit as FT      # noqa: E402
import fit129 as F9   # noqa: E402
import final129 as FN # noqa: E402
import menu as MN     # noqa: E402


def main(specs, specs_th, results, docsdir, menu, obsf, outdir):
    M = MN.Menu.load(menu)
    base = [json.loads(l) for l in open(obsf)]; base = [o for o in base if o["label"] == "fit129"]
    Sth = {json.loads(l)["tag"]: json.loads(l) for l in open(specs_th)}
    S, R, D = F9.load_rows(specs, results, docsdir)
    _, Rt, Dt = F9.load_rows(specs_th, results, docsdir); D.update(Dt)
    th = []
    for (t, sl), r in sorted(Rt.items()):
        s = Sth[t]; a = dict(s["mix"]["levels"]); a[MN.HEAD] = s["w129"]["head"]; a[MN.EMB] = "base"
        se = F9.lse(D, [((t, sl), 1), (("w129-u-k2.5", sl), -1)])
        th.append({"tag": f"{t}@{sl}", "assign": a, "label": sl, "kld": r["kld"], "se": se, "kind": "TH",
                   "size": round(M.bpw(a) + FN.OFFSET0 * 8 / M.P, 4)})
    rep = {"measured": {o["tag"]: o["kld"] for o in th}}
    for p in ("u4", "u2.75"):
        rows = [dict(o, assign=dict(o["assign"], **{MN.HEAD: p})) for o in base]
        Mo, x, _, frep = FT.fit_obs(M, rows, {"beta": "0"})
        res = {o["tag"]: math.log(Mo.predict(x, o["assign"]) / o["kld"]) for o in th}
        rep[f"heldout_v2as_{p}"] = {"residual_log": res, "rmse": float(np.sqrt(np.mean(np.square(list(res.values())))))}
        print(f"[held out, V2 head as {p}] log(pred / meas): " + ", ".join(f"{k.split('@')[0][9:]} {v:+.3f}" for k, v in res.items())
              + f" | RMSE {rep[f'heldout_v2as_{p}']['rmse']:.3f}")
    rows = [dict(o, assign=dict(o["assign"], **{MN.HEAD: "u2.75"})) for o in base] + th
    # the head's theta starts where the trunk's does (fit.py x0): its cost (err = the plan's head-only KL, ~1e-3) is then
    # ~1e-9 and its gradient vanishes, so it never moves. Start it at s_head = 0.5 (the measured head deltas are ~0.5x
    # hcurve's head-only KL deltas at both trunks) -- HEAD_INIT env, "0" = fit.py's x0.
    hi = float(os.environ.get("HEAD_INIT", 0.5))
    Mo0 = FT.build(M, rows, {"beta": "0"}); x0 = Mo0.x0()
    if hi > 0: x0[Mo0.layout()["theta"].stop - 1] = math.log(hi)
    Mo = FT.build(M, rows, {"beta": "0"}); Mo.layout()                 # fit.py's fit() reads self.N before layout() sets it
    x = Mo.fit(xi=x0); draws = Mo.posterior(x, 16, 169)
    pr = Mo.predict_rows(x, list(range(len(rows)))); km = np.array([o["kld"] for o in rows]); lr = np.log(pr / km)
    sev = np.array([o["se"] for o in rows])
    frep = {"rows": len(rows), "labels": Mo.NS, "params": Mo.N, "rmse_log": float(np.sqrt(np.mean(lr ** 2))),
            "chi2_per_row": float(np.mean(lr ** 2 / (sev ** 2 + Mo.tau ** 2))), "cost": float(Mo.res.cost), "nfev": int(Mo.res.nfev)}
    rep["head_init"] = hi; rep["s_head"] = float(Mo.tables(x)["s"][Mo.D.W.index(MN.HEAD)])
    res = {o["tag"]: math.log(Mo.predict(x, o["assign"]) / o["kld"]) for o in th}
    inter = {}
    for h in ("u3", "u4", "u5"):
        a3, a4 = f"w129-th-k3-h{h}@fit129", f"w129-th-k4-h{h}@fit129"
        by = {o["tag"]: o for o in th}
        inter[h] = {"meas": math.log(by[a3]["kld"] / by[a4]["kld"]), "pred": math.log(Mo.predict(x, by[a3]["assign"]) / Mo.predict(x, by[a4]["assign"]))}
    rep["refit"] = {"report": frep, "residual_log": res, "interaction_log_k3_over_k4": inter}
    print(f"[refit on 84 + 6, head init {hi}] s_head {rep['s_head']:.3g} {frep}; TH residuals " + ", ".join(f"{k.split('@')[0][9:]} {v:+.3f}" for k, v in res.items()))
    print("  log KL(K3) - log KL(K4) per head: " + ", ".join(f"{h} meas {v['meas']:.3f} pred {v['pred']:.3f}" for h, v in inter.items()))
    sl = FN.slopes(Mo, x, M, [1.5 + 0.25 * i for i in range(23)]); rep["refit"]["slopes"] = sl
    print("  slopes b: " + ", ".join(f"{k} {v['b']:.2f}" for k, v in sl.items()))
    out = {"report": frep, "model": Mo.dump(x), "x": x.tolist(), "draws": [v.tolist() for v in draws], "obs_tags": [o["tag"] for o in rows],
           "slopes": sl, "w129": {"menu": menu, "beta_mode": "0", "rows": "84 fit129 rows (V2 head as R16 u2.75) + 6 trunk x R16 head rows"}}
    json.dump(out, open(f"{outdir}/fit129h2.out.json", "w"))
    with open(f"{outdir}/fit129h2.obs.jsonl", "w") as f:
        for o in rows: f.write(json.dumps(o) + "\n")
    json.dump(rep, open(f"{outdir}/th129.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:8]))
