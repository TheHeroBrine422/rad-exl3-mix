# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fin133.py -- #173 follow-up (lead notes 11-13 10:15 / 10:50 AM): the final knapsack re-solve with the winning model
(M1) on the FULL ladder menu. Codebook-agnostic: everything comes from the menu (mkmenu.py over any ladder) and the
row files, so a hash ladder reruns the same commands.

  python3 fin133.py menu  OUT.json TRUNK_MENU.json [--emb-menu MENU174.json]
        merged menu: trunk + head options of TRUNK_MENU, the embedding options of --emb-menu (w135's menu174)
  python3 fin133.py fit   OUT_FIT.json MENU.json CANDS MODEL --ds DS.json.gz [--obs EXTRA.obs.jsonl ..] [--cut N]
        the model fitted on every search row of DS (ri <= cut) + DS's prior rows + extra obs files (w135's cal174);
        search / extra labels with parts ("S|p|..") are split per size when the model has relabel "size"
  python3 fin133.py solve OUT.json MENU.json CANDS MODEL FIT.json --ds DS.json.gz [--obs ..] --sizes 1,1.25,..
        [--state STATE --assign DIR] [--init DIR] [--bytes-off B] [--tol 0.02] [--save-dir DIR]
        per size: moves.mckp from every start (the size's incumbent, w129's init pick, the uniform level just below
        the size with the base head / embedding), the lowest predicted calib-scale KL inside the byte window wins;
        reports the pick's bpw, its head / embedding, level histogram, bytes moved vs incumbent / init, and predicted
        log ratios (c, x) vs incumbent / init; picks as assignment JSON (--save-dir)
  python3 fin133.py curve OUT.json MENU.json CANDS MODEL FIT.json --ds .. [--obs ..]
        the model's uniform-level KL (every trunk level, base head / embedding): the pricing of new levels (eighth
        steps should interpolate; sub-1 is extrapolation of the compounding)"""
import argparse, collections, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import ev133 as E             # noqa: E402
import m133 as MM             # noqa: E402
import moves as MV            # noqa: E402

MN = MM.MN


def merge_menu(trunk, embm):
    a = json.load(open(trunk))
    if embm:
        b = json.load(open(embm)); eb = [t for t in b["tensors"] if t["w"] == MN.EMB][0]
        a["tensors"] = [t if t["w"] != MN.EMB else eb for t in a["tensors"]]
        a.setdefault("built", {})["emb_menu"] = embm
    return a


def rows(ds, menu, extra, cut, relabel):
    """fit rows as fit.py obs dicts (assignments resolved against `menu`; the DS stores option indices of its own menu)"""
    W = ds.d["W"]; Md = ds.M
    def asg(t):
        a = {w: Md.T[Md.WI[w]]["opts"][j]["o"] for w, j in zip(W, ds.A[t])}
        if MN.EMB in menu.WI and MN.EMB not in a: a[MN.EMB] = "base"
        return a
    out = []
    for o in [o for o in ds.d["obs"] if o["ri"] <= cut] + ds.d["prior"]:
        out.append(dict(tag=o["tag"], assign=asg(o["a"]), label=o["label"], kld=o["kld"], se=o["se"], size=o.get("size")))
    for p in extra or []:
        for l in open(p):
            o = json.loads(l); out.append(dict(tag=o["tag"], assign=o["assign"], label=o["label"], kld=o["kld"], se=o.get("se", 0.0), size=o.get("size")))
    if relabel == "size":
        for o in out:
            if "|" in o["label"] and o.get("size") is not None: o["label"] = o["label"].replace("|", f"@{float(o['size']):g}|", 1)
    for o in out:                                                       # every menu tensor assigned (head / emb defaults)
        for t in menu.T:
            if t["w"] not in o["assign"]: o["assign"][t["w"]] = "base" if any(x["o"] == "base" for x in t["opts"]) else t["opts"][0]["o"]
    return out


def build(menu, obs, c):
    return MM.build(menu, obs, c.get("opts", {"tau": 0.02, "beta": "0", "xscale": True}), c.get("ext"))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("out"); ap.add_argument("menu")
    ap.add_argument("rest", nargs="*"); ap.add_argument("--emb-menu"); ap.add_argument("--ds"); ap.add_argument("--obs", nargs="*", default=[])
    ap.add_argument("--cut", type=int, default=10 ** 9); ap.add_argument("--sizes"); ap.add_argument("--state"); ap.add_argument("--assign")
    ap.add_argument("--init"); ap.add_argument("--bytes-off", type=int, default=8255936); ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--save-dir"); ap.add_argument("--min-k", type=float, default=None,
                    help="trunk options below this K are not offered (the model has no rows there: sub-1 is extrapolation)")
    ap.add_argument("--tag", default="pick133")
    a = ap.parse_args(argv)
    if a.cmd == "menu":
        m = merge_menu(a.menu, a.emb_menu); json.dump(m, open(a.out, "w"))
        M = MN.Menu(m); print(f"{a.out}: {len(M.T)} tensors, levels {len(M.levels())}, emb {[o['o'] for o in M.T[M.WI[MN.EMB]]['opts']]}, "
                              f"head {len(M.T[M.WI[MN.HEAD]]['opts'])}"); return 0
    M = MN.Menu.load(a.menu); cands = json.load(open(a.rest[0])); name = a.rest[1]; c = cands["models"][name]
    ds = E.DS(a.ds); obs = rows(ds, M, a.obs, a.cut, c.get("relabel"))
    Mo = build(M, obs, c)
    if a.cmd == "fit":
        x = Mo.fit(); pr = Mo.predict_rows(x, list(range(len(obs)))); km = np.array([o["kld"] for o in obs])
        rep = {"rows": len(obs), "params": int(Mo.N), "cost": float(Mo.res.cost), "rmse_log": float(np.sqrt(np.mean(np.log(pr / km) ** 2))),
               "lam": Mo.lam(x), "emb": dict(zip(Mo.levels, Mo.unpack(x)["emb"].tolist())), "emb_names": Mo.D.emb_names}
        json.dump({"model": name, "menu": a.menu, "obs_tags": [o["tag"] for o in obs], "report": rep, "x": x.tolist()}, open(a.out, "w"))
        print(json.dumps(rep)); return 0
    F = json.load(open(a.rest[2])); Mo.layout(); x = np.array(F["x"]); assert len(x) == Mo.N, (len(x), Mo.N)
    M.fixed = M.fixed + a.bytes_off
    if a.cmd == "curve":
        out = {}
        for lv in M.levels():
            u = M.uniform(lv, "u4" if any(o["o"] == "u4" for o in M.T[M.WI[MN.HEAD]]["opts"]) else None, "base")
            out[lv] = {"bpw": M.bpw(u), "c": float(np.exp(Mo.logpart(x, [u], "c")[0])), "K": M.opt(M.W[0], lv)["K"]}
            print(f"{lv:28s} {out[lv]['bpw']:.3f} bpw  pred c {out[lv]['c']:.5f}")
        json.dump(out, open(a.out, "w"), indent=1); return 0
    # solve
    PR = MV.Pred(Mo, x)
    if a.min_k is not None:                                             # disallowed options: +10 linearised log cost
        bad = [np.array([(o.get("K") is not None and t["w"] not in (MN.HEAD, MN.EMB) and o["K"] < a.min_k) for o in t["opts"]])
               for t in (M.T[M.WI[w]] for w in Mo.D.W)]
        lin0 = PR.lin
        def lin(asg):
            J, dl = lin0(asg)
            return J, [np.where(b, 10.0, d) for b, d in zip(bad, dl)]          # +10 log: never worth it (inf breaks mckp's lambda scale)
        PR.lin = lin
    inc = {}
    if a.state:
        for l in open(a.state):
            e = json.loads(l)
            if e["ev"] == "incumbent": inc[e["size"]] = e["tag"]
    def full(asg):
        asg = dict(asg)
        for t in M.T:
            if t["w"] not in asg: asg[t["w"]] = "base" if any(o["o"] == "base" for o in t["opts"]) else t["opts"][0]["o"]
        return asg
    lv = M.levels(); res = {"min_k": a.min_k, "model": name, "fit": a.rest[2], "menu": a.menu, "bytes_off": a.bytes_off, "sizes": {}}
    for size in [float(s) for s in a.sizes.split(",")]:
        hi = size * M.P / 8; lo = (size - a.tol) * M.P / 8
        starts = {}
        if size in inc and a.assign: starts["incumbent"] = full(json.load(open(f"{a.assign}/{inc[size]}.json")))
        ip = f"{a.init}/init-{size:g}.json" if a.init else None
        if ip and os.path.exists(ip): starts["init"] = full(json.load(open(ip)))
        hd = "u4" if any(o["o"] == "u4" for o in M.T[M.WI[MN.HEAD]]["opts"]) else None
        under = [n for n in lv if M.bytes(M.uniform(n, hd, "base")) <= hi and (a.min_k is None or M.opt(M.W[0], n)["K"] >= a.min_k)]
        starts["uniform"] = M.uniform(under[-1] if under else lv[0], hd, "base")
        best = None; per = {}
        for k, s0 in starts.items():
            p = MV.mckp(PR, s0, hi, lo)
            if p is None: continue
            kl = PR.kl(p); inw = lo <= M.bytes(p) <= hi; per[k] = {"kl": kl, "inwin": inw, "bpw": M.bpw(p)}
            if best is None or (inw, -kl) > (best[2], -best[0]): best = (kl, p, inw, k)
        if best is None or not best[2]:
            res["sizes"][f"{size:g}"] = {"infeasible": True, "starts": per}
            print(f"size {size:g}: no pick inside the window (min_k {a.min_k})", flush=True); continue
        pick = best[1]
        def cmp(ref):
            ch = [w for w in ref if ref[w] != pick[w]]
            mb = sum(abs(M.opt(w, pick[w])["bytes"] - M.opt(w, ref[w])["bytes"]) for w in ch)
            lc = Mo.logpart(x, [pick, ref], "c"); lx = Mo.logpart(x, [pick, ref], "x")
            return {"changed": len(ch), "moved_MB": round(mb / 1e6, 1), "pred_c": float(lc[0] - lc[1]), "pred_x": float(lx[0] - lx[1])}
        hist = collections.Counter(M.opt(w, pick[w])["K"] for w in M.W if w not in (MN.HEAD, MN.EMB))
        r = {"from": best[3], "inwin": best[2], "bpw": M.bpw(pick), "pred_c_kl": best[0], "head": pick.get(MN.HEAD), "emb": pick.get(MN.EMB),
             "levels": {f"{k:g}": v for k, v in sorted(hist.items())}, "starts": per,
             "vs": {k: cmp(v) for k, v in starts.items() if k != "uniform"}}
        res["sizes"][f"{size:g}"] = r
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True); json.dump(pick, open(f"{a.save_dir}/{a.tag}-{size:g}.json", "w"))
        print(f"size {size:g}: {r['bpw']:.4f} bpw from {r['from']} pred c-KL {r['pred_c_kl']:.5f} head {r['head']} emb {r['emb']} | "
              + " ".join(f"vs {k}: ch {v['changed']} {v['moved_MB']} MB c {v['pred_c']:+.3f} x {v['pred_x']:+.3f}" for k, v in r["vs"].items())
              + f" | K range {min(hist)}-{max(hist)}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
