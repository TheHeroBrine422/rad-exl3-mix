#!/usr/bin/env python3
"""rep151h.py -- #177-03 (w151): the R16 head price table with the mad codebook, from score119 outputs (as rep140.py table).
hcurve151.json = hcurve140.json with every scored mad donor (R16s003mad-K<K>) as plan u<K> (codebook "mad"; mad replaces
mul1i8, user 11-13), the mul1 plans u<K>m unchanged, the i8 plans moved to "i8_plans" (the record), monotonicity flags per
codebook, and per K the head-only KL ratio mad / i8 on the fit and search dumps (calib-weighted KL, as the plans' "kl").
Env as rep140 (R119, SC, DOCS, SHARE, PERMS, OUT) plus H140 (hcurve140.json)."""
import json, os, re, sys

sys.path.insert(0, os.environ.get("R119", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../yaqa/w119")))
import report119 as R  # noqa: E402
import numpy as np     # noqa: E402


def row(S, n):
    k = S.kl(n); z = np.load(f"{R.SC}/{S.name}/{n}.npz"); mt = S.rare_token()
    return {"kl": S.mean(k), "kl_unweighted": S.mean(k, False), "top1": float(z["hit"].mean()),
            "rare_script": float(k[S.rare_doc[S.di]].mean()), "rare_token": float(k[mt].mean()), "n_rare_token": int(mt.sum())}


h = json.load(open(os.environ["H140"])); F, X = R.Set("fit"), R.Set("search")
h["i8_plans"] = {k: p for k, p in h["plans"].items() if p["codebook"] == "i8"}
h["plans"] = {k: p for k, p in h["plans"].items() if p["codebook"] != "i8"}
don = sorted((float(m.group(1)), m.group(0)[:-4]) for f in os.listdir(f"{R.SC}/fit")
             if (m := re.fullmatch(r"R16s003mad-K([\d.]+)\.npz", f)) and os.path.exists(f"{R.SC}/search/{f}"))
cmp = {}
for K, n in don:
    r = row(F, n)
    h["plans"][f"u{K:g}"] = {"avg_bits": K, "kl": r["kl"], "kl_unweighted": r["kl_unweighted"], "top1": r["top1"],
                             "guard_kl": r["rare_token"], "guard_n": r["n_rare_token"], "rare_script_kl": r["rare_script"],
                             "donor": n, "codebook": "mad"}
    i8 = h["i8_plans"].get(f"u{K:g}")
    if i8:
        rs = row(X, n); ri = row(X, i8["donor"])
        cmp[f"{K:g}"] = {"fit_kl_ratio": r["kl"] / i8["kl"], "search_kl_ratio": rs["kl"] / ri["kl"],
                         "fit_top1": [i8["top1"], r["top1"]], "i8_donor": i8["donor"]}
flags = []
for cb in ("mad", "mul1"):
    pts = sorted((p["avg_bits"], p["kl"], k) for k, p in h["plans"].items() if p["codebook"] == cb)
    flags += [f"{b[2]} KL {b[1]:.3e} above {a[2]} {a[1]:.3e}" for a, b in zip(pts, pts[1:]) if b[1] > a[1]]
h["recipe"] = h["recipe"] + "; u<K> = mad at K since #177-03 (w151; i8 rows in i8_plans)"
h["flags_w151"] = flags; h["mad_vs_i8"] = cmp
json.dump(h, open(f"{R.OUT}/hcurve151.json", "w"), indent=1)
g = [v["fit_kl_ratio"] for v in cmp.values()]
R.say(f"price table: {R.OUT}/hcurve151.json ({len(h['plans'])} plans: {len(don)} mad); mad / i8 fit KL geo "
      f"{float(np.exp(np.mean(np.log(g)))) if g else float('nan'):.4f} over {len(g)} K; flags: {flags or 'none'}")
for K, v in cmp.items(): print(f"  K{K}: fit {v['fit_kl_ratio']:.4f}  search {v['search_kl_ratio']:.4f}  top1 {v['fit_top1'][0]:.4f} -> {v['fit_top1'][1]:.4f}")
