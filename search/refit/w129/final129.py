#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""final129.py MENU OBS.jsonl BETA OUT.json [--head-map OLD=NEW] [--labels L1,L2] -- #169's refit for the search driver:
fit.py on the calibration rows (OBS = fit129.py's prior_obs129.jsonl, rows of the given labels only), the fit.py output
format the driver reloads ({report, model, x, draws, obs_tags} + "slopes"), the rows actually fitted next to it
(OUT minus .out.json + .obs.jsonl: the driver's CONF prior_obs), and the fitted per-size slopes b = -d log KLD / d bpw of
the model's knapsack optimum at size -/+ 0.1 bpw (moves.mckp; container bpw = menu + w126's 8,255,936 B offset).
--head-map base=u4: the calibration mixes carry V2 3.00's head (4.0 bits); on the R16-head menu (mkmenu --heads) it is
priced as the R16 plan of the same bits (an approximation: one constant head in every row, so mostly the label offset)."""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import fit as FT    # noqa: E402
import menu as MN   # noqa: E402
import moves as MV  # noqa: E402

OFFSET0 = 8255936


def slopes(Mo, x, M, sizes):
    pred = MV.Pred(Mo, x); off = OFFSET0 * 8 / M.P; out = {}
    h = M.T[M.WI[MN.HEAD]]["opts"]; e = M.T[M.WI[MN.EMB]]["opts"][0]["o"]
    for size in sizes:
        pts = []
        for s2 in (size - 0.1, size + 0.1):
            hi = (s2 - off) * M.P / 8; lo = hi - 0.02 * M.P / 8
            best = None
            for ho in h:                                     # start from the largest uniform level under s2, any head
                for n in M.levels():
                    a = M.uniform(n, ho["o"], e)
                    if M.bpw(a) + off <= s2 and (best is None or M.bytes(a) > M.bytes(best)): best = a
            if best is None: continue
            m = MV.mckp(pred, best, hi, lo)
            if m is not None: pts.append((M.bpw(m) + off, math.log(pred.kl(m))))
        if len(pts) == 2 and pts[1][0] - pts[0][0] > 1e-4:
            out[f"{size:g}"] = {"b": -(pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0]), "kl_pred": math.exp(0.5 * (pts[0][1] + pts[1][1]))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("menu"); ap.add_argument("obs"); ap.add_argument("beta"); ap.add_argument("out")
    ap.add_argument("--head-map", action="append", default=[]); ap.add_argument("--labels", default="fit129")
    a = ap.parse_args(argv)
    M = MN.Menu.load(a.menu); hm = dict(x.split("=", 1) for x in a.head_map); labs = set(a.labels.split(","))
    obs = [json.loads(l) for l in open(a.obs)]; obs = [o for o in obs if o["label"] in labs]
    for o in obs:
        if o["assign"].get(MN.HEAD) in hm: o["assign"][MN.HEAD] = hm[o["assign"][MN.HEAD]]
    Mo, x, draws, rep = FT.fit_obs(M, obs, {"beta": a.beta}, ndraw=16, seed=169)
    sl = slopes(Mo, x, M, [1.5 + 0.25 * i for i in range(23)])
    out = {"report": rep, "model": Mo.dump(x), "x": x.tolist(), "draws": [v.tolist() for v in draws], "obs_tags": [o["tag"] for o in obs],
           "slopes": sl, "w129": {"menu": a.menu, "beta_mode": a.beta, "labels": sorted(labs), "head_map": hm}}
    json.dump(out, open(a.out + ".part", "w")); os.replace(a.out + ".part", a.out)
    ob = a.out[:-len(".out.json")] + ".obs.jsonl" if a.out.endswith(".out.json") else a.out + ".obs.jsonl"
    with open(ob, "w") as f:
        for o in obs: f.write(json.dumps(o) + "\n")
    print(json.dumps(rep)); print("slopes b: " + ", ".join(f"{k} {v['b']:.2f}" for k, v in sl.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
