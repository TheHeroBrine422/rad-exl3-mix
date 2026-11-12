#!/usr/bin/env python3
"""rep122.py -- w122 readouts on score119 outputs (RULES.md): scoring identity vs w119, E5 guards on the 17 donors, the
fit-gen price table (headsim-style hcurve122.json for v3/levels/reprice/headgrid.py).
Env as report119 (SC = scores root with <set>/ref.npz, <set>/<option>.npz; DOCS, DOCS94, SHARE, PERMS, OUT), plus W119
(w119's score dirs: W119/<set>/ref.npz, R16s003*-K*.npz, S-K*.npz)."""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("R119", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../w119")))
import report119 as R

W119, OUT = os.environ["W119"], R.OUT
GRID = [1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5]
OPTS = [(k, f"R16s003-K{k:g}") for k in GRID] + [(7, "R16s003m1-K7")]
SK = [3, 4, 5, 6, 6.5, 7]                                            # w119's S donors (σ 0.1, Fisher-sorted)


def wkl(name, st):
    return np.load(f"{W119}/{st}/{name}.npz")["kl"].astype(np.float64)


def ident(st):
    a, b = np.load(f"{R.SC}/{st}/ref.npz"), np.load(f"{W119}/{st}/ref.npz")
    ref = {k: bool(np.array_equal(a[k], b[k])) for k in b.files}
    sc = {}
    for mine, theirs in (("repro-K4", "R16s003-K4"), ("R16s003-K4", "R16s003-K4"), ("R16s003-K6", "R16s003-K6")):
        f = f"{R.SC}/{st}/{mine}.npz"
        if os.path.exists(f): sc[f"{mine} vs w119 {theirs}"] = bool(np.array_equal(np.load(f)["kl"], np.load(f"{W119}/{st}/{theirs}.npz")["kl"]))
    return {"ref": ref, "scores": sc}


def guards(S, k):
    mr, mt = S.rare_doc[S.di], S.rare_token()
    return float(k[mr].mean()), float(k[mt].mean()), int(mr.sum()), int(mt.sum())


def sinterp(gs, K):
    """S guard at K: measured, or log-linear in K between the neighbouring S donors (None outside 3..7)"""
    if K in gs: return gs[K]
    lo = [k for k in SK if k < K]; hi = [k for k in SK if k > K]
    if not lo or not hi: return None
    a, b = lo[-1], hi[0]; f = (K - a) / (b - a)
    return tuple(math.exp((1 - f) * math.log(gs[a][i]) + f * math.log(gs[b][i])) for i in range(2))


def main():
    res = {"ident": {}, "sets": {}}
    for st in ("search", "fit"):
        res["ident"][st] = ident(st); R.say(f"IDENT {st}: {res['ident'][st]}")
        S = R.Set(st); rows = {}
        gs = {}
        for k in SK:
            kk = wkl(f"S-K{k:g}", st); g = guards(S, kk); gs[k] = g[:2]
        prev = None
        for K, n in OPTS:
            k = S.kl(n); z = np.load(f"{R.SC}/{st}/{n}.npz")
            g = guards(S, k); si = sinterp(gs, K)
            r = {"K": K, "codebook": "mul1" if "m1" in n else "i8", "kl": S.mean(k), "kl_unweighted": S.mean(k, False),
                 "top1": float(z["hit"].mean()), "rare_script": g[0], "rare_token": g[1], "n_rare_script": g[2], "n_rare_token": g[3],
                 "S_rare_script": si[0] if si else None, "S_rare_token": si[1] if si else None, "S_interp": K not in gs}
            r["ratio_rare_script"] = g[0] / si[0] if si else None
            r["ratio_rare_token"] = g[1] / si[1] if si else None
            flags = []
            if si and r["ratio_rare_script"] > 1.10: flags.append("rare-script > 1.10x S")
            if si and r["ratio_rare_token"] > 1.30: flags.append("rare-token > 1.30x S")
            r["above_lower_K"] = bool(prev and g[0] > prev[0] and g[1] > prev[1])
            r["flags"] = flags; prev = g; rows[n] = r
            R.say(f"{st} {n:15s} KL {r['kl']:.3e} (unw {r['kl_unweighted']:.3e}) top1 {r['top1']:.4f} rare-script {g[0]:.3e}"
                  f" rare-token {g[1]:.3e}" + (f" | vs S{'~' if r['S_interp'] else ''} {r['ratio_rare_script']:.3f} / {r['ratio_rare_token']:.3f}" if si else "")
                  + (f"  FLAG {flags}" if flags else ""))
        res["sets"][st] = rows
    for n in res["sets"]["fit"]:                     # RULES: guard KL above the lower-K neighbour's on both sets
        if all(res["sets"][st][n]["above_lower_K"] for st in ("search", "fit")):
            for st in ("search", "fit"): res["sets"][st][n]["flags"].append("guards above the lower-K donor on both sets")
            R.say(f"FLAG {n}: guards above the lower-K donor on both sets")
    R.say("E5 flags: " + str({n: res['sets']['search'][n]['flags'] + res['sets']['fit'][n]['flags'] for n in res['sets']['fit']
                               if res['sets']['search'][n]['flags'] or res['sets']['fit'][n]['flags']} or "none"))
    json.dump(res, open(f"{OUT}/e5.json", "w"), indent=1)
    F = res["sets"]["fit"]; S = R.Set("fit")
    plans = {f"u{r['K']:g}": {"avg_bits": float(r["K"]), "kl": r["kl"], "kl_unweighted": r["kl_unweighted"], "top1": r["top1"],
                             "guard_kl": r["rare_token"], "guard_n": r["n_rare_token"], "rare_script_kl": r["rare_script"],
                             "donor": n, "codebook": r["codebook"]} for n, r in F.items()}
    json.dump({"n": int(S.N), "guard_rows": 20000, "set": "w119 fit-gen dump (200 fit-pool docs), calib-weighted KL",
               "recipe": "R16 order, two-sided LDLQ, sigma_out 0.03, f16 activations (w119 U); u7 = K7 mul1", "plans": plans},
              open(f"{OUT}/hcurve122.json", "w"), indent=1)
    R.say(f"price table: {OUT}/hcurve122.json ({len(plans)} donors)")


if __name__ == "__main__":
    main()
