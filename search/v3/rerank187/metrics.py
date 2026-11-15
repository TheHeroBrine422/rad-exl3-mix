# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""metrics.py -- the RULES-187 replay table, paired bootstrap and winner (sections 3 and 5).

  metrics.py RUNDIR CANDS.jsonl OUTDIR [ARMS]     RUNDIR: the replay's output (arm/N-group.json); ARMS default a,b,c0.1,c0.3,c1.0[,d]
  metrics.py base RUNDIR CANDS.jsonl              prints the best of a, b, c* by S (the base for arm d)

Per arm and band: rho (Spearman, whole-slice rows), slope, overstatement, hit (early-stopped rows censored), audit rho,
per-chunk rho; S = mean(rho_b2, rho_b3); paired bootstrap (rows resampled within band, 2000 draws, seed 187) of
dS = S(arm) - S(a) (80 % CI) and of d rho_b1. Winner: highest S among the arms, passing (i) dS >= 0.10,
(ii) rho_b1(arm) >= rho_b1(a) - 0.05, (iii) the 80 % CI of dS has a lower bound > 0. Else a."""
import csv, glob, json, math, os, sys

import numpy as np
from scipy.stats import spearmanr

CHUNK = 20
BANDS = ["b1", "b2", "b3", "b4"]
MARGIN, B1_TOL, NBOOT, SEED = 0.10, 0.05, 2000, 187
DEFAULT_ARMS = ["a", "b", "c0.1", "c0.3", "c1.0"]


def load_cands(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [c for c in rows if c["result_idx"] > CHUNK]


def chunk_of(idx): return CHUNK * ((idx - 1) // CHUNK)


def load_preds(rundir, arm, cands):
    """tag -> prediction for arm (NaN where the job did not produce one)"""
    out = {}
    for c in cands:
        N = chunk_of(c["result_idx"]); grp = c["band"] if arm == "a" else "all"
        p = os.path.join(rundir, arm, f"{N:04d}-{grp}.json")
        if not os.path.exists(p): out[c["tag"]] = float("nan"); continue
        d = json.load(open(p)); v = d["preds"].get(c["tag"])
        out[c["tag"]] = float("nan") if v is None or not math.isfinite(v) else float(v)
    return out


def _sp(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0: return float("nan")
    return float(spearmanr(x, y).correlation)


def band_metrics(rows, arm):
    """rows: dicts with band, stopped, audit, meas, pred (this arm), chunk (N)"""
    out = {}
    for b in BANDS + ["all"]:
        R = [r for r in rows if (b == "all" or r["band"] == b) and math.isfinite(r["pred"])]
        W = [r for r in R if not r["stopped"]]
        x = np.array([r["pred"] for r in W]); y = np.array([r["meas"] for r in W])
        m = {"n_whole": len(W), "n_all": len(R), "rho": _sp(x, y)}
        m["slope"] = float(np.polyfit(x, y, 1)[0]) if len(W) >= 3 and np.ptp(x) > 0 else float("nan")
        neg = x < 0
        m["overstatement"] = float(np.mean(x[neg]) / np.mean(y[neg])) if neg.sum() >= 2 and np.mean(y[neg]) != 0 else float("nan")
        pn = [r for r in R if r["pred"] < 0]
        m["hit"] = (sum(1 for r in pn if not r["stopped"] and r["meas"] < 0) / len(pn)) if pn else float("nan")
        m["n_pred_better"] = len(pn)
        xc = np.clip(x, -1.0, 1.0)                                     # robust view: a log ratio of 1 (2.7x) is the clip
        m["slope_clip"] = float(np.polyfit(xc, y, 1)[0]) if len(W) >= 3 and np.ptp(xc) > 0 else float("nan")
        m["overstatement_clip"] = float(np.mean(xc[xc < 0]) / np.mean(y[xc < 0])) if (xc < 0).sum() >= 2 and np.mean(y[xc < 0]) != 0 else float("nan")
        m["n_extreme"] = int(np.sum(np.abs(np.array([r["pred"] for r in R])) > 5))   # predictions beyond |log r| 5 (floor 1e-9 region)
        m["base_rate"] = (sum(1 for r in R if not r["stopped"] and r["meas"] < 0) / len(R)) if R else float("nan")
        ch = []
        for N in sorted({r["chunk"] for r in W}):
            cw = [r for r in W if r["chunk"] == N]
            if len(cw) >= 3: ch.append(_sp(np.array([r["pred"] for r in cw]), np.array([r["meas"] for r in cw])))
        ch = [v for v in ch if math.isfinite(v)]
        m["chunk_rho_mean"] = float(np.mean(ch)) if ch else float("nan"); m["chunk_rho_n"] = len(ch)   # chunks with >= 3 whole rows
        A = [r for r in W if r["audit"]]
        m["audit_rho"] = _sp(np.array([r["pred"] for r in A]), np.array([r["meas"] for r in A])) if len(A) >= 3 else float("nan")
        m["audit_n"] = len(A)
        out[b] = m
    out["S"] = float(np.mean([out["b2"]["rho"], out["b3"]["rho"]]))
    return out


def boot_draws(rows_by_arm, arm, ref="a", seed=SEED):
    """paired bootstrap over whole-slice rows within each band: dS = S(arm) - S(ref), rho_b1(arm) - rho_b1(ref)"""
    rng = np.random.default_rng(seed)
    base = {}
    for b in BANDS:
        W = [r for r in rows_by_arm[ref] if r["band"] == b and not r["stopped"] and math.isfinite(r["pred"])]
        V = {r["tag"]: r for r in rows_by_arm[arm] if r["band"] == b and not r["stopped"] and math.isfinite(r["pred"])}
        pairs = [(r["pred"], V[r["tag"]]["pred"], r["meas"]) for r in W if r["tag"] in V]
        base[b] = np.array(pairs) if pairs else np.zeros((0, 3))
    dS, db1 = [], []
    for _ in range(NBOOT):
        rho = {}
        for b, P in base.items():
            if len(P) < 3: rho[b] = (float("nan"), float("nan")); continue
            ix = rng.integers(0, len(P), len(P)); Q = P[ix]
            rho[b] = (_sp(Q[:, 0], Q[:, 2]), _sp(Q[:, 1], Q[:, 2]))
        dS.append(np.nanmean([rho["b2"][1], rho["b3"][1]]) - np.nanmean([rho["b2"][0], rho["b3"][0]]))
        db1.append(rho["b1"][1] - rho["b1"][0])
    dS, db1 = np.array(dS), np.array(db1)
    return {"dS_mean": float(np.nanmean(dS)), "dS_ci80": [float(np.nanpercentile(dS, 10)), float(np.nanpercentile(dS, 90))],
            "drho_b1_ci80": [float(np.nanpercentile(db1, 10)), float(np.nanpercentile(db1, 90))]}


def winner(table, boots):
    arms = [a for a in boots]                                       # the candidate arms (not "a", not "driver_live")
    ok = []
    for a in arms:
        dS = table[a]["S"] - table["a"]["S"]; b1 = table[a]["b1"]["rho"] - table["a"]["b1"]["rho"]
        bo = boots[a]
        reasons = {"dS": dS, "b1_ok": bool(b1 >= -B1_TOL), "ci_ok": bool(bo["dS_ci80"][0] > 0), "dS_ok": bool(dS >= MARGIN)}
        reasons["pass"] = reasons["dS_ok"] and reasons["b1_ok"] and reasons["ci_ok"]
        if reasons["pass"]: ok.append((table[a]["S"], a))
        table[a]["_gate"] = reasons
    if not ok: return "a", table
    return max(ok)[1], table


def main(argv):
    if argv[0] == "base":
        rundir, cpath = argv[1], argv[2]
        cands = load_cands(cpath); rows_arms = {}
        best = None
        for arm in ["a", "b", "c0.1", "c0.3", "c1.0"]:
            P = load_preds(rundir, arm, cands)
            R = [dict(_row(c), pred=P[c["tag"]], arm=arm) for c in cands]
            S = band_metrics(R, arm)["S"]
            if math.isfinite(S) and (best is None or S > best[0]): best = (S, arm)
        print(best[1]); return 0
    rundir, cpath, outdir = argv[0], argv[1], argv[2]
    arms = argv[3].split(",") if len(argv) > 3 else DEFAULT_ARMS
    cands = load_cands(cpath); os.makedirs(outdir, exist_ok=True)
    rows_by_arm, table = {}, {}
    live = [dict(_row(c), pred=c["pred_live"]) for c in cands]
    for arm in arms:
        P = load_preds(rundir, arm, cands) if arm != "d" else load_preds(rundir, "d", cands)
        rows_by_arm[arm] = [dict(_row(c), pred=P[c["tag"]]) for c in cands]
        table[arm] = band_metrics(rows_by_arm[arm], arm)
    table["driver_live"] = band_metrics(live, "driver_live")
    boots = {a: boot_draws(rows_by_arm, a) for a in arms if a != "a" and a in rows_by_arm}
    win, table = winner(table, boots)
    # reproduction: replay arm a vs the driver's live pred_logr on the same candidates
    if "a" in rows_by_arm:
        pa = np.array([r["pred"] for r in rows_by_arm["a"]]); pl = np.array([r["pred"] for r in live])
        m = np.isfinite(pa)
        table["reproduction"] = {"n": int(m.sum()), "pearson": float(np.corrcoef(pa[m], pl[m])[0, 1]),
                                 "spearman": _sp(pa[m], pl[m]), "mean_abs_diff": float(np.mean(np.abs(pa[m] - pl[m])))}
    res = {"arms": arms, "winner": win, "margin": MARGIN, "b1_tol": B1_TOL, "boot": boots, "table": table,
           "rules": "RULES-187 section 5"}
    json.dump(res, open(os.path.join(outdir, "table.json"), "w"), indent=1, default=float)
    json.dump({"winner": win, "gates": {a: table[a].get("_gate") for a in arms if a in table and a != "a"},
               "S": {a: table[a]["S"] for a in arms if a in table}}, open(os.path.join(outdir, "winner.json"), "w"), indent=1)
    write_csv(os.path.join(outdir, "rows.csv"), rows_by_arm, live, arms)
    write_md(os.path.join(outdir, "table.md"), table, boots, arms, win)
    print(json.dumps({"winner": win, "S": {a: round(table[a]["S"], 3) for a in arms}}))
    return 0


def _row(c):
    return {"tag": c["tag"], "band": c["band"], "stopped": bool(c["stopped"]), "audit": bool(c["audit"]), "meas": c["logr"],
            "chunk": chunk_of(c["result_idx"]), "size": c["size"], "kind": c["kind"], "decision": c["decision"]}


def write_csv(path, rows_by_arm, live, arms):
    keys = ["tag", "chunk", "band", "size", "kind", "stopped", "audit", "decision", "meas"]
    cols = keys + ["pred_live"] + [f"pred_{a}" for a in arms if a in rows_by_arm]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        base = {r["tag"]: r for r in live}
        for i, r in enumerate(live):
            vals = [r.get(k) for k in keys] + [r["pred"]] + [rows_by_arm[a][i]["pred"] for a in arms if a in rows_by_arm]
            w.writerow(vals)


def fmt(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def write_md(path, table, boots, arms, win):
    L = ["# #187 replay table (RULES-187; generated by metrics.py, do not edit by hand)", "",
         f"Winner: **{win}** (rule: S = mean rho_b2, rho_b3; margin {MARGIN}; b1 tolerance {B1_TOL}; 80 % paired bootstrap CI of dS > 0).", "",
         "| arm | S | rho b1 | rho b2 | rho b3 | rho b4 | slope b2+b3 raw / clip ±1 | hit b2+b3 | overst. b2+b3 | audit rho (all) | chunk rho b2+b3 | dS vs a (80 % CI) | predictions beyond ±5 (b2+b3) | gate |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms + ["driver_live"]:
        if a not in table: continue
        t = table[a]; bb = [t["b2"], t["b3"]]
        slope = np.nanmean([x["slope"] for x in bb]); hit = np.nanmean([x["hit"] for x in bb])
        slc = np.nanmean([x["slope_clip"] for x in bb]); nex = sum(x["n_extreme"] for x in bb)
        ov = np.nanmean([x["overstatement"] for x in bb]); cr = np.nanmean([x["chunk_rho_mean"] for x in bb])
        dsd = boots.get(a); g = t.get("_gate")
        dstr = f"{g['dS']:+.3f} [{dsd['dS_ci80'][0]:+.3f}, {dsd['dS_ci80'][1]:+.3f}]" if dsd and g else "–"
        L.append(f"| {a} | {fmt(t['S'], 3)} | {fmt(t['b1']['rho'])} | {fmt(t['b2']['rho'])} | {fmt(t['b3']['rho'])} | {fmt(t['b4']['rho'])} | "
                 f"{fmt(slope)} / {fmt(slc)} | {fmt(hit)} | {fmt(ov, 1)} | {fmt(t['all']['audit_rho'])} | {fmt(cr)} | {dstr} | {nex} | "
                 f"{('pass' if g and g['pass'] else ('–' if not g else 'fail'))} |")
    L += ["", "n (whole-slice rows / all decisions with a prediction) per band, from the first arm:", ""]
    t0 = table[arms[0]]
    L.append("| band | whole-slice rows | all rows (early-stopped censored) | predicted-better rows | audit whole rows |")
    L.append("|---|---|---|---|---|")
    for b in BANDS + ["all"]:
        L.append(f"| {b} | {t0[b]['n_whole']} | {t0[b]['n_all']} | {t0[b]['n_pred_better']} | {t0[b]['audit_n']} |")
    if "reproduction" in table:
        r = table["reproduction"]
        L += ["", f"Reproduction (arm a replay vs the driver's live pred_logr, n={r['n']}): Pearson {r['pearson']:.3f}, "
              f"Spearman {r['spearman']:.3f}, mean |diff| {r['mean_abs_diff']:.4f}."]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
