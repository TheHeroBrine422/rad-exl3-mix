# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""boot193.py SNAP RDIR CURV.json OUTDIR [--sizes 2,2.5,3,3.5,4,4.5,5,6] [--B 200] [--make 2.5,3.5,...] [--commit C]
-- RULES-193 §5: bootstrap of the §2 consensus -> per-tensor K bands; "uncertain" tensors; +- probe sets.

Bootstrap: per draw, the source models of each size resampled with replacement and the two slices' source weights
scaled by a Dirichlet(1, 1) pair (s186e vs s190a). Band = 10-90 % of K_t(s). fit.py's K = my refit's sensitivity-optimal
allocation (curv193 `opt`). Uncertain at size s: band width >= 0.5, or |consensus K - fit.py K| >= 0.5 with fit.py's K
outside the band. Probe "+": the incumbent with each uncertain tensor moved toward fit.py's K (to it, at most 2 steps);
"-": the same tensors moved one step away from fit.py's K; bytes balanced outside the set (bal193)."""
import argparse, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import common193 as C  # noqa: E402
import cons193 as CS  # noqa: E402
import bal193 as BL  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("rdir"); ap.add_argument("curv"); ap.add_argument("out")
    ap.add_argument("--sizes", default="2,2.5,3,3.5,4,4.5,5,6"); ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--make", default=""); ap.add_argument("--commit", default="?")
    a = ap.parse_args()
    M = C.load_menu(a.snap); km = C.KM(M); st = C.load_state(a.snap); cu = json.load(open(a.curv))
    src = CS.sources(a.snap, st); inc = {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
    sizes = [float(x) for x in a.sizes.split(",")]
    rng = np.random.default_rng(193); by = {}
    for i, s in enumerate(src): by.setdefault(s[0], []).append(i)
    base = CS.Curve(km, src); T0 = {s: base.targets(s) for s in sizes}
    draws = {s: [] for s in sizes}
    for b in range(a.B):
        we, wa = rng.dirichlet([1, 1]) * 2
        pick = []
        for s, ix in by.items(): pick += list(rng.choice(ix, len(ix)))
        sb = [(src[i][0], src[i][1] * (we if src[i][3].startswith("s186e") else wa), src[i][2], src[i][3]) for i in pick]
        cv = CS.Curve(km, sb)
        for s in sizes: draws[s].append([cv.target(w, s) for w in km.W])
    res = {}
    for s in sizes:
        A = np.array(draws[s]); lo, hi = np.percentile(A, 10, 0), np.percentile(A, 90, 0)
        opt = cu["opt"].get(f"{s:g}"); fk = {w: km.Kof[w][opt[w]] for w in km.W} if opt else None
        ia = C.assign(a.snap, inc[s]); unc = []
        for t, w in enumerate(km.W):
            wide = hi[t] - lo[t] >= 0.5
            dis = fk is not None and abs(T0[s][w] - fk[w]) >= 0.5 and not (lo[t] <= fk[w] <= hi[t])
            if wide or dis:
                unc.append({"w": w, "lo": float(lo[t]), "hi": float(hi[t]), "cons": float(T0[s][w]), "fit": fk[w] if fk else None,
                            "inc": km.Kof[w][ia[w]], "wide": bool(wide), "dis": bool(dis)})
        res[f"{s:g}"] = {"n_unc": len(unc), "median_width": float(np.median(hi - lo)), "unc": unc}
        print(f"{s:g}: median band {np.median(hi - lo):.3f}, uncertain {len(unc)} (wide {sum(u['wide'] for u in unc)}, "
              f"disagree {sum(u['dis'] for u in unc)}); toward fit.py: up {sum(1 for u in unc if u['fit'] is not None and u['fit'] > u['inc'])} "
              f"down {sum(1 for u in unc if u['fit'] is not None and u['fit'] < u['inc'])}")
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/boot.json", "w"), indent=1)
    if a.make:
        import fitopt193 as FO, curv193 as CV
        cache = {}
        for s in [float(x) for x in a.make.split(",")]:
            g = CV.band(s)
            if g not in cache: cache[g] = FO.load(M, a.rdir, g)[0]
            pred = cache[g]; ia = C.assign(a.snap, inc[s]); U = [u for u in res[f"{s:g}"]["unc"] if u["fit"] is not None and u["fit"] != u["inc"]]
            for sign in ("+", "-"):
                b = dict(ia)
                for u in U:
                    d = 1 if u["fit"] > u["inc"] else -1
                    if sign == "+":
                        steps = d * min(2, max(1, round(abs(u["fit"] - u["inc"]) / 0.125)))
                    else: steps = -d
                    b[u["w"]] = BL.shift(M, b, u["w"], steps)
                bb = BL.balance(pred, M, b, {u["w"] for u in U}, s)
                if bb is None: print(f"{s:g}{sign}: no balance"); continue
                pm = math.log(pred.kl(bb) / pred.kl(ia))
                d = {"size": s, "mode": "probe", "assign": bb,
                     "note": f"#193 boot{sign} | {len(U)} uncertain tensors moved {'toward' if sign == '+' else 'away from'} fit.py's K: "
                             f"measures the allocation-side uncertainty | code {a.commit}"}
                why = C.check_file(M, d)
                print(f"{s:g}{sign}: {len(U)} tensors forced, {sum(bb[w] != ia[w] for w in km.W)} changed, bpw {M.bpw(bb):.4f}, fit.py log r {pm:+.4f} {why or 'OK'}")
                if why is None: json.dump(d, open(f"{a.out}/boot{'p' if sign == '+' else 'm'}-{s:g}.json", "w"))


if __name__ == "__main__":
    main()
