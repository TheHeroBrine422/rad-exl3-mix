# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rankreg193.py gen SNAP RDIR WORK [--n 40] [--sizes ...]   (VM)  candidates from moves.generate / generate_cross on my
                                                             fit.py refit, from the size's incumbent and pool members;
                                                             a snap-like WORK dir (assign/ = SNAP's + new) + cands.jsonl
   rankreg193.py score TRAIN_FEATS NEW_FEATS OUT.json         (colo, xgboost) the §4 reg arm trained on every allowed
                                                             stage-1 row (whole only, as RULES-187 arm d) -> scores
   rankreg193.py pick SNAP WORK OUT.json OUTDIR [--top 3 --commit C]   decide files for the top-scored per size
RULES-193 amendment 5 (rankreg)."""
import argparse, glob, hashlib, json, math, os, random, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
KINDS = ["pair", "block", "partial", "thompson", "explore", "solve"]


def gen(a):
    import common193 as C, fitopt193 as FO, curv193 as CV, moves as MV, cons193 as CS
    M = C.load_menu(a.snap); st = C.load_state(a.snap)
    inc, pools = {}, {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
        if e.get("ev") == "pool": pools[float(e["size"])] = e["members"]
    # amendment 13: the driver's current reference per size = the `ref` of its latest own submit on the active slice
    # (an identity; no score is read), else the incumbent
    act = None; lastref = {}
    for e in st:
        if e.get("ev") == "rotate": act = e["to"]; lastref = {}
        if e.get("ev") == "submit" and e.get("slice") == act and e.get("ref") and e.get("kind") in KINDS + ["cross", "manual"]:
            lastref[float(e["size"])] = e["ref"]
    seen = set()
    for p in glob.glob(f"{a.snap}/assign/*.json"): seen.add(MV.ahash(json.load(open(p))))
    os.makedirs(f"{a.work}/assign", exist_ok=True)
    for f in ("menu-v2.json", "state.jsonl", "results.jsonl"):
        if not os.path.exists(f"{a.work}/{f}"): os.symlink(os.path.abspath(f"{a.snap}/{f}"), f"{a.work}/{f}")
    for p in glob.glob(f"{a.snap}/assign/*.json"):
        q = f"{a.work}/assign/{os.path.basename(p)}"
        if not os.path.exists(q): os.symlink(os.path.abspath(p), q)
    rng = random.Random(a.seed); cache = {}; rows = []
    for s in [float(x) for x in a.sizes.split(",")]:
        g = CV.band(s)
        if g not in cache: cache[g] = FO.load(M, a.rdir, g, draws=True)
        pred, draws = cache[g]; lo, hi = C.window(M, s)
        mem = pools.get(s, [inc[s]])
        # the driver's reference: the worst member (largest s190a KLD) when the pool is full
        ref = lastref.get(s, inc[s])
        if not os.path.exists(f"{a.snap}/assign/{ref}.json"): ref = inc[s]
        n = 0; tries = 0
        while n < a.n and tries < 4 * a.n:
            tries += 1
            par = rng.choice(mem); pa = C.assign(a.snap, par)
            if rng.random() < 0.15 and len(mem) > 1:
                oth = C.assign(a.snap, rng.choice([m for m in mem if m != par]))
                c = MV.generate_cross(pred, pa, oth, lo, hi, seen, rng); kind = "cross"
            else:
                kind = KINDS[tries % len(KINDS)]; c = MV.generate(kind, pred, draws, pa, lo, hi, seen, rng)
            if c is None: continue
            seen.add(MV.ahash(c)); tag = f"rr{a.seed}-{s:g}-{kind}-{n:03d}"
            json.dump(c, open(f"{a.work}/assign/{tag}.json", "w"))
            ra = C.assign(a.snap, ref)
            rows.append({"tag": tag, "size": s, "band": "b1" if s <= 2.5 else ("b2" if s <= 4 else "b3"), "kind": kind,
                         "slice": act or "live", "ref": ref, "result_idx": 10 ** 6, "stopped": False, "audit": False,
                         "decision": None, "logr": 0.0, "pred_live": math.log(pred.kl(c) / pred.kl(ra)), "parent": par})
            n += 1
        print(f"{s:g}: {n} candidates (ref {ref})", flush=True)
    with open(f"{a.work}/cands.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


def score(a):
    import xgboost as xgb
    import rank193 as R
    T = json.load(open(a.train)); Nw = json.load(open(a.new)); names = T["names"]
    tr = [c for c in T["cands"] if not c["stopped"] and math.isfinite(T["F"][c["tag"]]["base_pred"])]
    X = np.array([[T["F"][c["tag"]][n] for n in names] for c in tr]); y = np.array([c["logr"] for c in tr])
    m = xgb.XGBRegressor(objective="reg:squarederror", **R.XGB_COMMON); m.fit(X, y)
    Xn = np.array([[Nw["F"][c["tag"]][n] for n in names] for c in Nw["cands"]])
    out = {c["tag"]: float(v) for c, v in zip(Nw["cands"], m.predict(Xn))}
    json.dump({"n_train": len(tr), "scores": out}, open(a.out, "w")); print(len(tr), "train rows;", len(out), "scored")


def pick(a):
    import common193 as C
    M = C.load_menu(a.snap); S = json.load(open(a.scores))["scores"]
    cands = [json.loads(l) for l in open(f"{a.work}/cands.jsonl")]
    os.makedirs(a.outdir, exist_ok=True); corr = {}
    if a.anom:
        import anom193 as A
        km = C.KM(M); G, X = A.design([{"cand": c["tag"], "refa": c["ref"]} for c in cands], a.work, km, M)
        res = json.load(open(a.anom))["ii"]["groups"]
        for j, c in enumerate(cands):
            b = C.BAND[c["size"]]
            corr[c["tag"]] = float(sum(res[g][b]["beta"] * X[j, i] for i, g in enumerate(G)
                                       if res[g][b]["se"] and abs(res[g][b]["beta"] / res[g][b]["se"]) >= 2))
    for s in sorted({c["size"] for c in cands}):
        L = sorted([c for c in cands if c["size"] == s], key=lambda c: S[c["tag"]])
        for k, c in enumerate(L[:a.top]):
            asg = json.load(open(f"{a.work}/assign/{c['tag']}.json"))
            d = {"size": s, "mode": "decide", "assign": asg,
                 "note": f"#193 rankreg | xgboost regressor (RULES-187 arm d + curve features) ranks this {c['kind']} move "
                         f"top {k + 1} of {len(L)} at {s:g} (score {S[c['tag']]:+.4f}, refit pred {c['pred_live']:+.4f}) | code {a.commit}"}
            why = C.check_file(M, d)
            print(f"{s:g} #{k + 1}: {c['tag']} score {S[c['tag']]:+.4f} refit pred {c['pred_live']:+.4f} {why or 'OK'}")
            if why is None: json.dump(d, open(f"{a.outdir}/rankreg-{s:g}-{k + 1}.json", "w"))
        if a.anom:                                                    # amendment 13: rankanom = score + anomaly correction
            got = {c["tag"] for c in L[:a.top]}
            L2 = sorted([c for c in L if c["tag"] not in got], key=lambda c: S[c["tag"]] + corr[c["tag"]])
            for k, c in enumerate(L2[:a.top2]):
                asg = json.load(open(f"{a.work}/assign/{c['tag']}.json"))
                d = {"size": s, "mode": "decide", "assign": asg,
                     "note": f"#193 rankanom | xgboost score {S[c['tag']]:+.4f} + cut-A anomaly correction {corr[c['tag']]:+.4f} "
                             f"(version ii, |z| >= 2 groups) ranks this {c['kind']} move top {k + 1} of the rest at {s:g} | code {a.commit}"}
                why = C.check_file(M, d)
                print(f"{s:g} anom #{k + 1}: {c['tag']} score {S[c['tag']]:+.4f} corr {corr[c['tag']]:+.4f} {why or 'OK'}")
                if why is None: json.dump(d, open(f"{a.outdir}/rankanom-{s:g}-{k + 1}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="*")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--sizes", default="2,2.5,3,3.5,4,4.5,5,6")
    ap.add_argument("--top", type=int, default=3); ap.add_argument("--commit", default="?")
    ap.add_argument("--seed", type=int, default=193); ap.add_argument("--anom"); ap.add_argument("--top2", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "gen": a.snap, a.rdir, a.work = a.args; gen(a)
    elif a.cmd == "score": a.train, a.new, a.out = a.args; score(a)
    elif a.cmd == "pick": a.snap, a.work, a.scores, a.outdir = a.args; pick(a)
