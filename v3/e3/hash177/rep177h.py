#!/usr/bin/env python3
"""rep177h.py -- RULES-177h readout: per (arm, K, set) the calib-weighted head-only KL ratio arm / I (w122's mul1i8 donor
R16s003-K<K>) with report119's document bootstrap, unweighted ratio, top-1 delta, rare-script / rare-token guards, and the
equal-quality delta-bpw = K - K_eq (negative = the arm saves bits) on w122's 17-donor i8 curve (KL log-linear in K between grid neighbours). Gate line: gate-i8-K4's
KL vector vs R16s003-K4's. Env as report119 (SC = our scores root, DOCS, SHARE, PERMS, OUT) plus HSC (w122's scores root).
(w139, #177)"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("R119", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../yaqa/w119")))
import report119 as R

HSC = os.environ["HSC"]
GRID = [1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5]
ARMS, KS = ["H", "HG", "HC"] + [a for a in os.environ.get("ARMS_EXTRA", "").split(",") if a], [2.5, 4, 6]


def curve(S):
    ks, kl = [], []
    for k in GRID:
        f = f"{HSC}/{S.name}/R16s003-K{k:g}.npz"
        if os.path.exists(f):
            ks.append(k); kl.append(S.mean(np.load(f)["kl"].astype(np.float64)))
    return np.array(ks), np.log(np.array(kl))


def k_equal(ks, lkl, x):
    """the K at which the i8 curve has KL x (log-linear between neighbours; KL falls with K)"""
    lx = math.log(x)
    for i in range(len(ks) - 1):
        a, b = lkl[i], lkl[i + 1]
        if (a - lx) * (b - lx) <= 0 and a != b: return ks[i] + (ks[i + 1] - ks[i]) * (a - lx) / (a - b)
    return float("nan")


def main():
    res = {}
    for st in ("fit", "search"):
        S = R.Set(st); ks, lkl = curve(S); hit = lambda o: float(np.load(f"{R.SC}/{st}/{o}.npz")["hit"].mean())
        if S.has("gate-i8-K4"):
            same = bool(np.array_equal(np.load(f"{R.SC}/{st}/gate-i8-K4.npz")["kl"], np.load(f"{HSC}/{st}/R16s003-K4.npz")["kl"]))
            R.say(f"GATE {st}: gate-i8-K4 KL vector {'== R16s003-K4 (SAME)' if same else 'DIFFERS (FAIL)'}"); res[f"gate_{st}"] = same
        for a in ARMS:
            for K in KS:
                o, c = f"{a}-K{K:g}", f"R16s003-K{K:g}"
                if not (S.has(o) and S.has(c)): continue
                ka, kb = S.kl(c), S.kl(o); r = S.ratio(ka, kb); ru = S.ratio(ka, kb, False); g = S.guard(ka, kb)
                m = S.mean(kb); ke = k_equal(ks, lkl, m)
                row = {"ratio": r, "unweighted": ru, "kl": m, "kl_i8": S.mean(ka), "top1": hit(o), "top1_i8": hit(c),
                       "rare_script": g[0], "rare_token": g[1], "k_equal": ke, "dbpw": K - ke if ke == ke else None}   # RULES-177h amendment 1
                res.setdefault(st, {}).setdefault(a, {})[f"{K:g}"] = row
                lab = "better" if r[2] < 1 else "costs" if r[1] > 1 else "at par"
                R.say(f"{st} {a} K{K:g}: R {R.fmt(r)} ({lab}); unweighted {R.fmt(ru)}; KL {m:.4e} vs {S.mean(ka):.4e}; "
                      f"top-1 {100 * row['top1']:.2f} vs {100 * row['top1_i8']:.2f}; rare-script {g[0]:.3f} rare-token {g[1]:.3f}; "
                      f"K_eq {ke:.3f} (dbpw {K - ke:+.3f})")
    # amendment 2: seed floor (i8 s1 / s0, s2 / s0) and the 3-seed ratio R3 = sum_seeds KL(H) / sum_seeds KL(i8)
    for st in ("fit", "search"):
        S = R.Set(st)
        for K in (2.5, 4):
            i8 = [f"R16s003-K{K:g}"] + [f"i8s{j}-K{K:g}" for j in (1, 2)]; hh = [f"H-K{K:g}"] + [f"Hs{j}-K{K:g}" for j in (1, 2)]
            i8, hh = [o for o in i8 if S.has(o)], [o for o in hh if S.has(o)]
            if len(i8) < 2: continue
            fl = {o: S.ratio(S.kl(i8[0]), S.kl(o)) for o in i8[1:]}
            row = {"floor": fl, "n_i8": len(i8), "n_h": len(hh)}
            if hh:
                r3 = S.ratio(sum(S.kl(o) for o in i8) / len(i8), sum(S.kl(o) for o in hh) / len(hh)); row["R3"] = r3
            res.setdefault("amend2", {}).setdefault(st, {})[f"{K:g}"] = row
            R.say(f"A2 {st} K{K:g}: seed floor " + ", ".join(f"{o.split('-')[0]}/s0 {R.fmt(v)}" for o, v in fl.items())
                  + (f"; R3 H/i8 ({len(hh)} / {len(i8)} seeds) {R.fmt(row['R3'])}" if "R3" in row else ""))
    a2 = res.get("amend2", {})
    for K in ("2.5", "4"):
        if all(K in a2.get(st, {}) and "R3" in a2[st][K] for st in ("fit", "search")):
            f, s_ = a2["fit"][K]["R3"], a2["search"][K]["R3"]
            v = "resolved: hash costs" if f[1] > 1 and s_[1] > 1 else "resolved: hash better" if f[2] < 1 and s_[2] < 1 else "within noise"
            R.say(f"A2 K{K}: {v} (fit {R.fmt(f)}, search {R.fmt(s_)})")
    json.dump(res, open(f"{R.OUT}/rep177h.json", "w"), indent=1)


main()
