# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sub2.py -- RULES-floor202 amendment 3 (lead, 2026-11-16): fitsat on the sub-2 sizes only.

Rows of a b1 fit input are kept when their size is in SIZES (labels without a kept anchor row are dropped). Arms per fit
point: L = the live b1 fit.py output (what the driver used), a = fit.py on the kept rows, s = fit202 sat on the kept rows
(started from a). Candidates are w188's cands-b1.json rows of those sizes (stage 1, predicted by the last fit before them).

  sub2.py filt IN.json OUT.json SIZES
  sub2.py pred CANDS.json FITDIR K OUTDIR SIZES        (colo; one fit point; skipped when OUTDIR/<K>.json exists)
  sub2.py table CANDS.json OUTDIR SIZES                 per size: Spearman(pred, measured log r) on whole-slice rows,
                                                        pair accuracy (same size x slice x reference, all rows); bootstrap
                                                        (rows within size, 2000, seed 202) 80 % CI of s - L and s - a
  sub2.py live IN.json OUTDIR SIZES                     fit a and s on the kept rows of IN, report, write s's out.json"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def sz(a): return [float(x) for x in a.split(",")]


def filt(d, sizes):
    keep = [o for o in d["obs"] if o.get("size") is not None and round(float(o["size"]), 4) in sizes]
    anc = {o["label"] for o in keep if o.get("se", 1) == 0}
    keep = [o for o in keep if o["label"] in anc]
    return dict(d, obs=keep)


def fits(d):
    import fit202 as F2, menu as MN
    M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
    opts = d.get("opts") or {}
    Ma, xa, _, ra = F2.fit_obs(M, d["obs"], opts, None)
    Ms, xs, ds, rs = F2.fit_obs(M, d["obs"], dict(opts, sat=True), list(xa), 16, 202)
    return M, (Ma, xa, ra), (Ms, xs, ds, rs)


def pred(cpath, fitdir, k, outdir, sizes):
    import fit as FT, menu as MN, moves as MV
    C = json.load(open(cpath)); f = C["fits"][k]; name = f["name"]
    outp = os.path.join(outdir, f"{k:03d}.json")
    if os.path.exists(outp): return
    te = [c for c in C["cands"] if c["fit"] == k and c["size"] in sizes]
    if not te: json.dump({"k": k, "preds": {}}, open(outp, "w")); return
    d0 = json.load(open(f"{fitdir}/{name}.in.json")); live = json.load(open(f"{fitdir}/{name}.out.json"))
    M = MN.Menu(d0["menu"]) if isinstance(d0["menu"], dict) else MN.Menu.load(d0["menu"])
    ML = FT.build(M, d0["obs"], d0.get("opts") or {}); ML.layout(); xL = np.array(live["x"])
    d = filt(d0, sizes); _, (Ma, xa, ra), (Ms, xs, _, rs) = fits(d)
    P = {"L": MV.Pred(ML, xL), "a": MV.Pred(Ma, xa), "s": MV.Pred(Ms, xs)}
    seen = {t.split("@")[0] for t in live["obs_tags"]}
    out = {arm: {} for arm in P}
    for c in te:
        if c["tag"] in seen: continue
        for arm in P: out[arm][c["tag"]] = math.log(P[arm].kl(C["assign"][c["tag"]]) / P[arm].kl(C["assign"][c["ref"]]))
    res = {"k": k, "fit": name, "rows": len(d["obs"]), "preds": out, "S": rs["S"], "logS_se": rs["logS_se"], "rep_a": ra, "rep_s": rs}
    json.dump(res, open(outp + ".part", "w")); os.replace(outp + ".part", outp)
    print(name, "rows", len(d["obs"]), "cands", len(te), "S %.3g" % rs["S"], "cost a %.1f s %.1f" % (ra["cost"], rs["cost"]))


def _rho(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation) if len(x) > 2 else float("nan")


def table(cpath, outdir, sizes):
    C = json.load(open(cpath)); P = {"L": {}, "a": {}, "s": {}}; Sv = []
    for fn in sorted(os.listdir(outdir)):
        if not fn.endswith(".json"): continue
        r = json.load(open(os.path.join(outdir, fn)))
        for arm in P: P[arm].update(r["preds"].get(arm, {}))
        if r.get("S"): Sv.append((r["k"], r["S"], r["rows"], r["rep_a"]["cost"], r["rep_s"]["cost"]))
    rows = [c for c in C["cands"] if c["size"] in sizes and c["tag"] in P["s"]]
    rng = np.random.default_rng(202)

    def met(rs):
        out = {}
        whole = [c for c in rs if not c["stopped"]]
        for arm in P: out["rho_" + arm] = _rho([P[arm][c["tag"]] for c in whole], [c["logr"] for c in whole])
        grp = {}
        for c in rs: grp.setdefault((c["slice"], c["ref"]), []).append(c)
        for arm in P:
            n = ok = 0
            for g in grp.values():
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        dm = g[i]["logr"] - g[j]["logr"]; dp = P[arm][g[i]["tag"]] - P[arm][g[j]["tag"]]
                        if dm == 0: continue
                        n += 1; ok += (dm > 0) == (dp > 0)
            out["pair_" + arm] = ok / n if n else float("nan")
        return out
    print("fit points", len(Sv), "S per point (k, S, rows, cost a, cost s):", [(k, round(s, 3), n, round(a, 1), round(b, 1)) for k, s, n, a, b in Sv])
    for s in sizes + [None]:
        rs = [c for c in rows if s is None or c["size"] == s]
        m = met(rs); bs = {"s-L": [], "s-a": []}
        for _ in range(2000):
            idx = rng.integers(0, len(rs), len(rs)); mb = met([rs[i] for i in idx])
            bs["s-L"].append(mb["rho_s"] - mb["rho_L"]); bs["s-a"].append(mb["rho_s"] - mb["rho_a"])
        ci = {k: [round(float(np.nanpercentile(v, 10)), 3), round(float(np.nanpercentile(v, 90)), 3)] for k, v in bs.items()}
        print("all" if s is None else s, "n", len(rs), "whole", sum(not c["stopped"] for c in rs), {k: round(v, 3) for k, v in m.items()}, "rho CI80", ci)


def live(inp, outdir, sizes):
    d = filt(json.load(open(inp)), sizes); os.makedirs(outdir, exist_ok=True)
    M, (Ma, xa, ra), (Ms, xs, ds, rs) = fits(d)
    tag = "-".join(f"{s:g}" for s in sizes)
    json.dump(d, open(f"{outdir}/sub2-{tag}.in.json", "w"))
    json.dump({"report": rs, "model": Ms.dump(xs), "x": xs.tolist(), "draws": [v.tolist() for v in ds], "obs_tags": [o["tag"] for o in d["obs"]],
               "opts": dict(d.get("opts") or {}, sat=True)}, open(f"{outdir}/sub2-{tag}.out.json", "w"))
    print("rows", len(d["obs"]), "plain", json.dumps(ra)); print("sat", json.dumps(rs))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "pred": pred(a[1], a[2], int(a[3]), a[4], sz(a[5]))
    elif a[0] == "table": table(a[1], a[2], sz(a[3]))
    elif a[0] == "live": live(a[1], a[2], sz(a[3]))
