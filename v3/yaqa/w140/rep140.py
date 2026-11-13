#!/usr/bin/env python3
"""rep140.py -- w140 readouts on score119 outputs (RULES.md): price rows for the mul1 donors K6 / 6.5 / 7.5 / 8 (keys u<K>m)
added to hcurve122.json's plans -> hcurve140.json, i8 vs mul1 at K6 / K6.5 (doc bootstrap), the curve to u8, the
monotonicity flag. Env as rep122 / report119 (SC, DOCS, SHARE, PERMS, OUT), plus H122 (hcurve122.json)."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("R119", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../w119")))
import report119 as R

OUT = R.OUT
NEW = [6, 6.5, 7.5, 8]
I8 = [5, 5.5, 6, 6.5]


def row(S, n):
    k = S.kl(n); z = np.load(f"{R.SC}/{S.name}/{n}.npz")
    mr, mt = S.rare_doc[S.di], S.rare_token()
    return {"kl": S.mean(k), "kl_unweighted": S.mean(k, False), "top1": float(z["hit"].mean()),
            "rare_script": float(k[mr].mean()), "rare_token": float(k[mt].mean()), "n_rare_script": int(mr.sum()), "n_rare_token": int(mt.sum())}


def main():
    res = {"sets": {}, "cmp": {}, "flags": []}
    for st in ("fit", "search"):
        S = R.Set(st); rows = {}
        for K in I8: rows[f"R16s003-K{K:g}"] = row(S, f"R16s003-K{K:g}") | {"K": K, "codebook": "i8"}
        for K in [7] + NEW: rows[f"R16s003m1-K{K:g}"] = row(S, f"R16s003m1-K{K:g}") | {"K": K, "codebook": "mul1"}
        for n, r in rows.items():
            R.say(f"w140 {st} {n:15s} KL {r['kl']:.3e} (unw {r['kl_unweighted']:.3e}) top1 {r['top1']:.4f} "
                  f"rare-script {r['rare_script']:.3e} rare-token {r['rare_token']:.3e}")
        res["sets"][st] = rows
        cmp = {}
        for K in (6, 6.5):
            a, b = S.kl(f"R16s003-K{K:g}"), S.kl(f"R16s003m1-K{K:g}")
            cmp[f"K{K:g}"] = {"ratio": S.ratio(a, b), "unweighted": S.ratio(a, b, False), "guards": S.guard(a, b)}
            c = cmp[f"K{K:g}"]
            R.say(f"w140 {st} mul1/i8 K{K:g}: {R.fmt(c['ratio'])} (unw {R.fmt(c['unweighted'])}); rare-script {c['guards'][0]:.3f}"
                  f" rare-token {c['guards'][1]:.3f}")
        for K, L in ((6.5, 6), (7, 6.5), (7.5, 7), (8, 7.5)):        # step factors along the mul1 curve
            a, b = S.kl(f"R16s003m1-K{L:g}"), S.kl(f"R16s003m1-K{K:g}")
            cmp[f"step m{L:g}->m{K:g}"] = {"ratio": S.ratio(a, b)}
            R.say(f"w140 {st} step m{L:g} -> m{K:g}: {R.fmt(cmp[f'step m{L:g}->m{K:g}']['ratio'])}")
        res["cmp"][st] = cmp
    verdict = {}
    for K in ("K6", "K6.5"):
        lo = [res["cmp"][st][K]["ratio"][1] for st in ("fit", "search")]; hi = [res["cmp"][st][K]["ratio"][2] for st in ("fit", "search")]
        verdict[K] = "mul1 better" if max(hi) < 1 else ("i8 better" if min(lo) > 1 else "tie")
        R.say(f"w140 VERDICT {K}: {verdict[K]}")
    res["verdict"] = verdict
    for K, L in ((6.5, 6), (7, 6.5), (7.5, 7), (8, 7.5)):            # RULES flag: above the next-lower-K mul1 donor on both dumps
        a, b = f"R16s003m1-K{L:g}", f"R16s003m1-K{K:g}"
        for m in ("kl", "rare_script", "rare_token"):
            if all(res["sets"][st][b][m] > res["sets"][st][a][m] for st in ("fit", "search")):
                res["flags"].append(f"{b} {m} above {a} on both dumps")
    R.say(f"w140 flags: {res['flags'] or 'none'}")
    json.dump(res, open(f"{OUT}/rep140.json", "w"), indent=1)
    h = json.load(open(os.environ["H122"])); F = res["sets"]["fit"]
    for K in NEW:
        r = F[f"R16s003m1-K{K:g}"]
        h["plans"][f"u{K:g}m"] = {"avg_bits": float(K), "kl": r["kl"], "kl_unweighted": r["kl_unweighted"], "top1": r["top1"],
                                 "guard_kl": r["rare_token"], "guard_n": r["n_rare_token"], "rare_script_kl": r["rare_script"],
                                 "donor": f"R16s003m1-K{K:g}", "codebook": "mul1"}
    h["recipe"] += "; u<K>m = mul1 at K (w140, #162-02)"
    json.dump(h, open(f"{OUT}/hcurve140.json", "w"), indent=1)
    R.say(f"price table: {OUT}/hcurve140.json ({len(h['plans'])} plans)")


if __name__ == "__main__":
    main()
