#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""inter173.py SPECS.jsonl DOCSUMS.json OUT.json [--menu MENU --models MODELS.json --name NAME --fit FIT.json --obs OBS..] --
(w158, RULES-173-03b) the interaction table: per pair r_a, r_b, r_ab (log KL vs the base on golden-i173c's documents), I = r_ab -
r_a - r_b, rho = I / (r_a + r_b), paired per-document bootstrap CIs (2,000, seed 1733, one resample set per size); with a fit:
M1m's predicted r_a, r_b, r_ab, I and the residual R = r_ab - r_ab(M1m). Per category, per size and pooled (inverse-variance
weighted mean of rho; CI from the same bootstrap resamples). Reading per RULES-173-03b."""
import argparse, collections, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NB, SEED = 2000, 1733


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("specs"); ap.add_argument("docs"); ap.add_argument("out")
    ap.add_argument("--menu"); ap.add_argument("--models"); ap.add_argument("--name"); ap.add_argument("--fit"); ap.add_argument("--obs", nargs="*")
    a = ap.parse_args(argv)
    S = {}
    for l in open(a.specs):
        if l.strip(): s = json.loads(l); S[s["tag"]] = s
    R = json.load(open(a.docs)); D = {t: np.array(v["docs"], float) for t, v in R.items()}
    n0 = next(iter(D.values()))[:, 1]
    for t, d in D.items(): assert np.array_equal(d[:, 1], n0), t
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(n0), size=(NB, len(n0)))
    pred = None
    if a.fit:
        import fin158 as F8, m158 as M8, menu as MN
        M = MN.Menu.load(a.menu); spec = json.load(open(a.models))[a.name]; obs = F8.load_obs(a.obs)
        Mo = F8.build(M, obs, spec); Mo.layout(); x = np.array(json.load(open(a.fit))["x"])
        pred = lambda asg: float(Mo.logpart(x, [asg], "c", mode="f16")[0])     # log KL without the offset (A8 = reference mode)
    lk = lambda t: math.log(D[t][:, 0].sum() / D[t][:, 1].sum())
    bk = lambda t: np.log(D[t][:, 0][idx].sum(1) / D[t][:, 1][idx].sum(1))
    out = {"pairs": [], "missing": sorted(t for t in S if t not in D)}
    for t, s in S.items():
        m = s["meta"]
        if m["arm"] != "ab": continue
        bt = f"w158i-b{m['size']:g}-base"; ta, tb, tab = (t[:-2] + "a", t[:-2] + "b", t)
        if not all(k in D for k in (bt, ta, tb, tab)): continue
        ra, rb, rab = lk(ta) - lk(bt), lk(tb) - lk(bt), lk(tab) - lk(bt)
        BA, BB, BAB, B0 = bk(ta), bk(tb), bk(tab), bk(bt)
        Ib = (BAB - B0) - (BA - B0) - (BB - B0); I = rab - ra - rb
        rhob = Ib / ((BA - B0) + (BB - B0))
        e = {"size": m["size"], "pair": m["pair"], "category": m["category"], "a": m["a"], "b": m["b"], "move_to": m["move_to"],
             "r_a": ra, "r_b": rb, "r_ab": rab, "I": I, "I_lo": float(np.percentile(Ib, 2.5)), "I_hi": float(np.percentile(Ib, 97.5)),
             "I_se": float(Ib.std()), "rho": I / (ra + rb), "rho_lo": float(np.percentile(rhob, 2.5)), "rho_hi": float(np.percentile(rhob, 97.5)),
             "rho_se": float(rhob.std()), "_rhob": rhob, "_Ib": Ib}
        if pred:
            p0, pa, pb, pab = (pred(S[k]["assign"]) for k in (bt, ta, tb, tab))
            e.update({"m_r_a": pa - p0, "m_r_b": pb - p0, "m_r_ab": pab - p0, "m_I": (pab - p0) - (pa - p0) - (pb - p0), "R": rab - (pab - p0)})
        out["pairs"].append(e)

    def pool(ps, tag):
        if not ps: return None
        w = np.array([1 / max(p["rho_se"], 1e-6) ** 2 for p in ps]); w = w / w.sum()
        rb = sum(wi * p["_rhob"] for wi, p in zip(w, ps)); r = float(sum(wi * p["rho"] for wi, p in zip(w, ps)))
        o = {"group": tag, "n": len(ps), "rho": r, "rho_lo": float(np.percentile(rb, 2.5)), "rho_hi": float(np.percentile(rb, 97.5)),
             "I_mean": float(np.mean([p["I"] for p in ps]))}
        if "R" in ps[0]:
            wi = np.array([1 / max(p["I_se"], 1e-6) ** 2 for p in ps]); wi = wi / wi.sum()
            o["absR_w"] = float(np.sum(wi * np.abs([p["R"] for p in ps]))); o["absI_w"] = float(np.sum(wi * np.abs([p["I"] for p in ps])))
        return o
    ps = out["pairs"]
    out["pooled"] = pool(ps, "all")
    out["by_size"] = [pool([p for p in ps if p["size"] == s], f"size {s:g}") for s in sorted({p["size"] for p in ps})]
    cat = lambda c: c.split("_early")[0].split("_mid")[0].split("_late")[0] if c.startswith(("ffn_pair", "gdn_input")) else ("attn_input" if c.startswith("attn_input") else c)
    out["by_category"] = [pool([p for p in ps if cat(p["category"]) == c], c) for c in sorted({cat(p["category"]) for p in ps})]
    po = out["pooled"]
    if po:
        add = po["rho_lo"] <= 0 <= po["rho_hi"] and abs(po["rho"]) < 0.10
        cats = [c["group"] for c in out["by_category"] if c and not (c["rho_lo"] <= 0 <= c["rho_hi"]) and abs(c["rho"]) >= 0.10]
        cap = ("absR_w" in po) and po["absR_w"] <= 0.5 * po["absI_w"]
        out["reading"] = {"additive_within_noise": bool(add), "interacting_categories": cats, "m1m_captures": bool(cap) if "absR_w" in po else None}
    for p in ps: p.pop("_rhob"); p.pop("_Ib")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(ps)} pairs ({len(out['missing'])} specs missing)")
    for p in ps:
        print(f"  {p['size']:g} {p['pair']:4s} {p['category']:20s} r_a {p['r_a']:+.4f} r_b {p['r_b']:+.4f} r_ab {p['r_ab']:+.4f} I {p['I']:+.4f} "
              f"[{p['I_lo']:+.4f}, {p['I_hi']:+.4f}] rho {p['rho']:+.3f}" + (f"  M1m I {p['m_I']:+.4f} R {p['R']:+.4f}" if "R" in p else ""))
    for g in [po] + out["by_size"] + out["by_category"]:
        if g: print(f"  {g['group']:22s} n {g['n']:2d} rho {g['rho']:+.3f} [{g['rho_lo']:+.3f}, {g['rho_hi']:+.3f}]" + (f" |R| {g['absR_w']:.4f} |I| {g['absI_w']:.4f}" if "absR_w" in g else ""))
    print("reading:", out.get("reading"))


if __name__ == "__main__":
    main()
