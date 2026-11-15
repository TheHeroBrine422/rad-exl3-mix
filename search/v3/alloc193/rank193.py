# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rank193.py -- RULES-193 §4: learn-to-rank (XGBRanker) vs fit.py in w165's time-ordered replay (RULES-187 metrics).

  rank193.py cands SNAP OUT.jsonl            candidate table of the full live state (stage-1 results on allowed slices;
                                             base = the live pre-submit pred_logr), w165's cands format
  rank193.py feats SNAP CANDS.jsonl OUT.json [--base-dir RUNDIR --base-arm a]
                                             features per candidate (w165 arm-d features + the incumbents' pooled curve
                                             deviation at the candidate's time); base prediction = the arm's replay
                                             prediction (w165 snapshot) or pred_live
  rank193.py replay FEATS.json OUTDIR        (colo, xgboost) arms: base, reg (w165 arm d: XGBRegressor on whole rows),
                                             rpair (XGBRanker rank:pairwise), rndcg (rank:ndcg, label = within-group rank
                                             bins); train at refit point N on rows with result index <= N (rankers incl.
                                             early-stopped rows), predict (N, N + 20]
  rank193.py table FEATS.json OUTDIR         RULES-187 table + paired bootstrap + winner rule vs base; within-group pairwise
                                             accuracy (groups = size x slice x reference)"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rerank187")); sys.path.insert(0, os.path.join(HERE, ".."))

KINDS = ["solve", "partial", "pair", "block", "thompson", "explore", "cross", "manual"]
GROUPS = ["attn_in", "attn_out", "ssm_in", "ssm_out", "ffn_gu", "ffn_down"]
GEN = set(KINDS)
CHUNK = 20
XGB_COMMON = dict(n_estimators=60, max_depth=2, learning_rate=0.1, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  random_state=193, n_jobs=1, tree_method="hist")


def band(s): return "b1" if s <= 2.5 else ("b2" if s <= 4.0 else ("b3" if s <= 6.0 else "b4"))


def cands(snap, out):
    import common193 as C
    st = C.load_state(snap); sub = {e["tag"]: e for e in st if e.get("ev") == "submit"}
    nres, n, res = {}, 0, {}
    for e in st:
        if e.get("ev") == "result": n += 1; nres.setdefault(e["tag"], n); res[e["tag"]] = e["r"]
    rows = []
    for e in st:
        if e.get("ev") not in ("decision", "tentative") or e.get("stage") == 2 or e.get("logr") is None: continue
        s = sub.get(e["tag"])
        if s is None or s["kind"] not in GEN or not C.ok_slice(s["slice"]) or e["tag"] not in nres: continue
        rows.append({"tag": e["tag"], "size": float(s["size"]), "band": band(float(s["size"])), "kind": s["kind"], "slice": s["slice"],
                     "ref": e.get("vs") or s.get("ref"), "result_idx": nres[e["tag"]], "stopped": bool(res[e["tag"]].get("stopped")),
                     "audit": bool(s.get("audit")), "decision": e.get("decision") or "tentative", "logr": e["logr"],
                     "pred_live": s.get("pred_logr")})
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(len(rows), "candidates; dropped", C.dropped)


def feats(snap, cpath, out, base_dir=None, base_arm="a"):
    import common193 as C, menu as MN
    M = C.load_menu(snap); km = C.KM(M); st = C.load_state(snap)
    cs = [json.loads(l) for l in open(cpath)]
    # incumbents over time (result index) -> pooled curve of the incumbents at that time
    inc_t, n = [], 0
    for e in st:
        if e.get("ev") == "result": n += 1
        if e.get("ev") == "incumbent": inc_t.append((n, float(e["size"]), e["tag"]))
    nper = {}
    for t in M.T:
        o = sorted(t["opts"], key=lambda x: x["bytes"]); nper[t["w"]] = 8 * (o[-1]["bytes"] - o[0]["bytes"]) / max(o[-1]["K"] - o[0]["K"], 1e-9)
    nv = np.array([nper[w] for w in km.W]); tot = nv.sum()
    tix = {w: i for i, w in enumerate(km.W)}
    typ = {w: (C.group_of(w)[0]) for w in km.W}; TY = C.TYPES + ["head", "emb"]
    curve_cache = {}

    def curve_at(N, s0):
        cur = {}
        for n_, s, t in inc_t:
            if n_ <= N: cur[s] = t
        key = (tuple(sorted(cur.items())), s0)
        if key not in curve_cache:
            src = [(s, 1.0, C.assign(snap, t), t) for s, t in cur.items()]
            if not src: curve_cache[key] = np.full(len(km.W), s0)
            else:
                cv = CSCurve(km, src); curve_cache[key] = np.array([cv.target(w, s0) for w in km.W])
        return curve_cache[key]
    import cons193 as CS
    CSCurve = CS.Curve
    base = {}
    if base_dir:
        for c in cs:
            N = CHUNK * ((c["result_idx"] - 1) // CHUNK); p = os.path.join(base_dir, base_arm, f"{N:04d}-{c['band']}.json")
            v = json.load(open(p))["preds"].get(c["tag"]) if os.path.exists(p) else None
            base[c["tag"]] = v if v is not None and math.isfinite(v) else float("nan")
    else:
        base = {c["tag"]: (c["pred_live"] if c.get("pred_live") is not None else float("nan")) for c in cs}
    nl = 64; F = {}; names = None
    for c in cs:
        cc, aa = C.assign(snap, c["tag"]), C.assign(snap, c["ref"])
        Kc = np.array([km.Kof[w][cc[w]] for w in km.W]); Ka = np.array([km.Kof[w][aa[w]] for w in km.W])
        ch = [i for i, w in enumerate(km.W) if cc[w] != aa[w] and w != MN.EMB]
        trunk = [i for i in ch if km.W[i] != MN.HEAD]
        layers = [M.T[M.WI[km.W[i]]]["layer"] for i in trunk]
        groups = {g: 0 for g in GROUPS}
        for i in trunk:
            g = M.T[M.WI[km.W[i]]]["group"]
            if g in groups: groups[g] += 1
        terc = [0, 0, 0]
        for l in layers: terc[min(2, 3 * l // nl)] += 1
        f = {"base_pred": base[c["tag"]]}
        for k in KINDS: f[f"kind_{k}"] = float(c["kind"] == k)
        f.update({"bpw_moved": 1000.0 * (M.bytes(cc) - M.bytes(aa)) * 8 / M.P, "n_changed": len(trunk),
                  "n_K_up": int(sum(Kc[i] > Ka[i] for i in trunk)), "n_K_down": int(sum(Kc[i] < Ka[i] for i in trunk)),
                  "mean_K_before": float(np.mean(Ka[trunk])) if trunk else 0.0, "mean_K_after": float(np.mean(Kc[trunk])) if trunk else 0.0,
                  "sum_K_delta": float(np.sum(Kc[trunk] - Ka[trunk])) if trunk else 0.0,
                  "mean_depth": float(np.mean(layers)) / nl if layers else 0.0})
        for g in GROUPS: f[f"n_{g}"] = groups[g]
        f.update({"n_layers_t1": terc[0], "n_layers_t2": terc[1], "n_layers_t3": terc[2],
                  "emb_changed": float(cc.get(MN.EMB) != aa.get(MN.EMB)), "head_changed": float(cc.get(MN.HEAD) != aa.get(MN.HEAD)),
                  "size": c["size"], "band": ["b1", "b2", "b3", "b4"].index(c["band"])})
        # §4 extra: distance to the incumbents' pooled curve (at the candidate's submit time ~ result index - outstanding)
        T = curve_at(c["result_idx"] - 22, c["size"])
        dc = np.abs(Kc - T) * nv / tot; da = np.abs(Ka - T) * nv / tot
        f["curve_dist_delta"] = float(dc.sum() - da.sum()); f["curve_dist"] = float(dc.sum())
        for ty in TY:
            m = np.array([typ[w] == ty for w in km.W]); f[f"curve_d_{ty}"] = float(dc[m].sum() - da[m].sum())
            f[f"bits_{ty}"] = float(((Kc - Ka) * nv / tot)[m].sum())
        F[c["tag"]] = f; names = list(f)
    json.dump({"names": names, "cands": cs, "F": F}, open(out, "w"))
    print(len(F), "candidates,", len(names), "features")


def replay(fpath, outdir):
    import xgboost as xgb
    d = json.load(open(fpath)); cs = d["cands"]; names = d["names"]; F = d["F"]
    cs = [c for c in cs if c["result_idx"] > CHUNK]
    X = {c["tag"]: np.array([F[c["tag"]][n] for n in names], float) for c in cs}
    def grp(c): return f"{c['size']}|{c['slice']}|{c['ref']}"
    os.makedirs(outdir, exist_ok=True); preds = {a: {} for a in ("base", "reg", "rpair", "rndcg")}
    maxN = max(c["result_idx"] for c in cs)
    for N in range(CHUNK, maxN + 1, CHUNK):
        te = [c for c in cs if N < c["result_idx"] <= N + CHUNK]
        if not te: continue
        for c in te: preds["base"][c["tag"]] = F[c["tag"]]["base_pred"]
        tr = [c for c in cs if c["result_idx"] <= N]
        trw = [c for c in tr if not c["stopped"]]
        Xte = np.vstack([X[c["tag"]] for c in te])
        if len(trw) >= 30:
            m = xgb.XGBRegressor(objective="reg:squarederror", **XGB_COMMON)
            m.fit(np.vstack([X[c["tag"]] for c in trw]), np.array([c["logr"] for c in trw]))
            for c, v in zip(te, m.predict(Xte)): preds["reg"][c["tag"]] = float(v)
        else:
            for c in te: preds["reg"][c["tag"]] = F[c["tag"]]["base_pred"]
        trs = sorted(tr, key=grp); gs = [grp(c) for c in trs]
        sizes_ = [gs.count(g) for g in dict.fromkeys(gs)]
        if len(trs) >= 30 and max(sizes_) >= 2:
            Xtr = np.vstack([X[c["tag"]] for c in trs])
            # pairwise: higher label = better = lower log r
            y = np.array([-c["logr"] for c in trs])
            m = xgb.XGBRanker(objective="rank:pairwise", **XGB_COMMON); m.fit(Xtr, y, group=sizes_)
            for c, v in zip(te, m.predict(Xte)): preds["rpair"][c["tag"]] = -float(v)
            # ndcg: integer relevance = within-group rank bins (0..4, best = 4)
            yl = np.zeros(len(trs)); i0 = 0
            for k in sizes_:
                seg = np.array([c["logr"] for c in trs[i0:i0 + k]]); r = seg.argsort().argsort()
                yl[i0:i0 + k] = np.floor(4.999 * (k - 1 - r) / max(k - 1, 1)); i0 += k
            m = xgb.XGBRanker(objective="rank:ndcg", **XGB_COMMON); m.fit(Xtr, yl, group=sizes_)
            for c, v in zip(te, m.predict(Xte)): preds["rndcg"][c["tag"]] = -float(v)
        else:
            for c in te: preds["rpair"][c["tag"]] = preds["rndcg"][c["tag"]] = F[c["tag"]]["base_pred"]
    json.dump(preds, open(os.path.join(outdir, "preds.json"), "w"))
    print({a: len(v) for a, v in preds.items()})


def table(fpath, outdir):
    import metrics as MT
    d = json.load(open(fpath)); cs = [c for c in d["cands"] if c["result_idx"] > CHUNK]
    P = json.load(open(os.path.join(outdir, "preds.json")))
    rows = {}
    for a, pr in P.items():
        rows[a] = [{"tag": c["tag"], "band": c["band"], "stopped": c["stopped"], "audit": c["audit"], "meas": c["logr"],
                    "pred": pr.get(c["tag"], float("nan")) if pr.get(c["tag"]) is not None else float("nan"),
                    "chunk": CHUNK * ((c["result_idx"] - 1) // CHUNK)} for c in cs]
    # the base may be NaN (no live prediction); compare all arms on rows where every arm has a finite prediction
    ok = {c["tag"] for c in cs if all(math.isfinite(rows[a][i]["pred"]) for a in rows for i in [cs.index(c)])}
    for a in rows: rows[a] = [r for r in rows[a] if r["tag"] in ok]
    rows["a"] = rows.pop("base")
    tab = {a: MT.band_metrics(rows[a], a) for a in rows}
    boots = {a: MT.boot_draws(rows, a, "a") for a in rows if a != "a"}
    win, tab = MT.winner(tab, boots)
    # within-group pairwise accuracy (whole rows, same size x slice x reference)
    def grp(c): return f"{c['size']}|{c['slice']}|{c['ref']}"
    byg = {}
    for c in cs:
        if c["tag"] in ok and not c["stopped"]: byg.setdefault(grp(c), []).append(c)
    pacc = {}
    for a in rows:
        pr = {r["tag"]: r["pred"] for r in rows[a]}; hit = tot = 0; hb = {}
        for g, L in byg.items():
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    dm = L[i]["logr"] - L[j]["logr"]; dp = pr[L[i]["tag"]] - pr[L[j]["tag"]]
                    if dm == 0 or dp == 0: continue
                    h = (dm > 0) == (dp > 0); hit += h; tot += 1
                    b = L[i]["band"]; hb.setdefault(b, [0, 0]); hb[b][0] += h; hb[b][1] += 1
        pacc[a] = {"all": hit / tot if tot else float("nan"), "n": tot, **{b: v[0] / v[1] for b, v in hb.items()}}
    res = {"table": tab, "boot": boots, "winner": win, "pair_acc": pacc, "n_rows": len(ok)}
    json.dump(res, open(os.path.join(outdir, "table.json"), "w"), indent=1, default=float)
    lines = [f"rows {len(ok)} (whole + stopped), winner by RULES-187 §5: **{win}** (a = base)", "",
             "| arm | S | rho b1 | rho b2 | rho b3 | n whole b1/b2/b3 | dS 80% CI | gate | pair acc (all / b1 / b2 / b3, n) |", "|---|---|---|---|---|---|---|---|---|"]
    for a in rows:
        t = tab[a]; g = t.get("_gate", {}); bo = boots.get(a, {})
        pa = pacc[a]
        lines.append(f"| {a} | {t['S']:.3f} | {t['b1']['rho']:.3f} | {t['b2']['rho']:.3f} | {t['b3']['rho']:.3f} | "
                     f"{t['b1']['n_whole']}/{t['b2']['n_whole']}/{t['b3']['n_whole']} | "
                     f"{'[%.3f, %.3f]' % tuple(bo['dS_ci80']) if bo else '-'} | {('pass' if g.get('pass') else 'fail') if g else '-'} | "
                     f"{pa['all']:.3f} / {pa.get('b1', float('nan')):.3f} / {pa.get('b2', float('nan')):.3f} / {pa.get('b3', float('nan')):.3f}, {pa['n']} |")
    open(os.path.join(outdir, "table.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "cands": cands(a[1], a[2])
    elif a[0] == "feats":
        bd = a[a.index("--base-dir") + 1] if "--base-dir" in a else None
        feats(a[1], a[2], a[3], bd)
    elif a[0] == "replay": replay(a[1], a[2])
    elif a[0] == "table": table(a[1], a[2])
