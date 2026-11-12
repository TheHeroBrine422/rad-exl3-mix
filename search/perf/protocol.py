#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""protocol.py ROWS.npz TIMING.json OUTDIR -- eval-slice sizing and the accept / reject rule for the V3 search (w123, #161).

ROWS.npz: rowpack.py's file (stored 27B score rows). TIMING.json: timing.py's cost model (seconds per prefilled token,
fixed seconds per eval / per extra radiance run). Writes OUTDIR/{noise.json, frac.json, decide.json, resample.json} and
prints the tables.

1 noise     per pair and part: the per-document noise s = SE_boot * sqrt(docs) of log r (r = pooled KL ratio B / A) and
            q = s / |log r| (noise per document relative to the effect). Pairs: two reports of different containers on
            the same golden with 0.5% <= |log r| <= 15%. The 3.00 -> 3.50 group moves (w123b2) are the level-move
            reference.
2 frac      two-part slices (27d / 27e / 27f, x + c from the same pair): SE of the pooled log r (KL-weighted over both
            parts, the readout's "both") when every part keeps a fraction f of its documents (draws without
            replacement, 300 per f), against SE(1) / sqrt(f) with the finite-population factor.
3 decide    normal model of one move: true log r = delta, per part d_p ~ N(delta, SE_p^2), SE_p = s_p / sqrt(n_p);
            pooled d ~ N(delta, SE^2). Priors delta ~ N(mu, sigma^2); per-document noise s (calib) with s_x = RX * s.
            Rules (accept = keep the move):
              fixed f        score the fraction f of the slice; accept iff d < 0 and d_x < 0 and d_c < 0 (same sign)
              seq f1 z1      score f1; accept iff z = d / SE <= -z1 and both parts < 0; reject iff z >= +z1;
                             else score the rest (to f = 1) and decide as fixed 1
            Value per card-hour: E[-delta * accept] (true KL gain banked), correct decisions (accept iff delta < 0),
            and wrong accepts (delta > 0 accepted), each per hour of one card at the measured cost.
4 resample  the same rules on real documents: each two-part pair's per-document KL sums with the per-document log
            ratios scaled to a target delta (B'_d = A_d (B_d / A_d)^(delta / log r)), slices drawn from the documents
            with replacement at the standard size; P(accept) vs the normal model's."""
import collections, itertools, json, math, os, re, sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slicese as S   # noqa: E402

INF = float("inf")
RULES = [("fixed", 1.0), ("fixed", 0.5), ("fixed", 0.25), ("fixed", 0.125),
         ("seq", (0.25, 1.0), (2.0,)), ("seq", (0.25, 1.0), (1.5,)), ("seq", (0.25, 1.0), (1.0,)),
         ("seq", (0.25, 1.0), ((2.0, 1.0),)), ("seq", (0.25, 1.0), ((2.0, 0.5),)), ("seq", (0.25, 1.0), ((1.5, 0.5),)),
         ("seq", (0.125, 0.5, 1.0), (1.5, 1.5)), ("seq", (0.125, 0.5, 1.0), ((2.0, 1.0), (2.0, 1.0))),
         ("seq", (0.25, 0.5, 1.0), ((2.0, 1.0), (2.0, 1.0))), ("seq", (0.125, 0.25, 0.5, 1.0), (2.0, 2.0, 2.0)),
         ("seq", (0.25, 1.0), ((2.0, 0.0),)), ("seq", (0.25, 1.0), ((2.5, 0.5),)), ("seq", (0.125, 1.0), ((2.0, 0.5),)),
         ("seq", (0.125, 0.5, 1.0), ((2.0, 0.5), (2.0, 0.5))), ("seq", (0.25, 0.5, 1.0), ((2.0, 0.5), (2.0, 0.5))),
         # reject early, accept only on the full slice (an accepted candidate is then the new incumbent, nothing left to score)
         ("seq", (0.25, 1.0), ((INF, 0.5),)), ("seq", (0.125, 0.5, 1.0), ((INF, 0.5), (INF, 0.5))),
         ("seq", (0.125, 0.5, 1.0), ((INF, 1.0), (INF, 1.0))), ("seq", (0.125, 0.5, 1.0), ((INF, 0.0), (INF, 0.5)))]
REC_RULE = ("seq", (0.125, 0.5, 1.0), ((INF, 0.0), (INF, 0.5)))      # the recommendation (section 3b)
TWO = ("golden-search27d", "golden-search27e", "golden-search27f")
RX = 1.0          # x-part per-document noise relative to calib (section 1 measures it; overridden below)


def boot_se(A, B, nb=1000, seed=0):
    rng = np.random.default_rng(seed); i = rng.integers(0, len(A), (nb, len(A)))
    return float(np.log(B[i].sum(1) / A[i].sum(1)).std())


def pairs_by_golden(G, IX):
    model = {(g, t): r["model"] for g, v in IX.items() for t, r in v["reports"].items()}
    for g, v in sorted(G.items()):
        for ta, tb in itertools.combinations(sorted(v["kl"]), 2):
            if model[(g, ta)] != model[(g, tb)]: yield g, ta, tb


def noise(G, IX):
    out = []
    for g, ta, tb in pairs_by_golden(G, IX):
        v = G[g]; A = S.doc_sums(v, v["kl"][ta])[0]; B = S.doc_sums(v, v["kl"][tb])[0]; ok = v["doc_n"] > 0; A, B = A[ok], B[ok]
        lr = math.log(B.sum() / A.sum())
        if not 0.005 <= abs(lr) <= 0.15: continue
        se = boot_se(A, B); s = se * math.sqrt(len(A))
        mv = ta.startswith("w123b2:") and tb.startswith("w123b2:")
        if mv and ("s27-all" in ta + tb or {ta[11:], tb[11:]} in ({"s27-L00-03", "s27-L00-07"}, {"s27-L56-63", "s27-L60-63"})): continue
        kind = ("move1" if "s27-base" in ta + tb else "move2") if mv else ("x" if g.endswith("-x") else "c")
        out.append({"golden": g, "a": ta, "b": tb, "kind": kind, "docs": int(len(A)), "logr": lr, "se": se, "s": s, "q": s / abs(lr),
                    "tok_per_doc": float(v["doc_tok"][ok].mean())})
    return out


def two_part(G):
    """(golden base, tag base) -> {"x": (A, B, tok), "c": (A, B, tok)} for pairs scored on both parts of 27d / e / f"""
    P = {}
    for g in TWO:
        vx, vc = G.get(g + "-x"), G.get(g + "-c")
        if vx is None or vc is None: continue
        base = lambda t: t[:-2] if t[-2:] in ("-x", "-c") else None
        tx = {base(t): t for t in vx["kl"] if base(t)}; tc = {base(t): t for t in vc["kl"] if base(t)}
        common = sorted(set(tx) & set(tc))
        for a, b in itertools.combinations(common, 2):
            rec = {}
            for p, v, T in (("x", vx, tx), ("c", vc, tc)):
                ok = v["doc_n"] > 0
                A = S.doc_sums(v, v["kl"][T[a]])[0][ok]; B = S.doc_sums(v, v["kl"][T[b]])[0][ok]
                if np.array_equal(A, B): break
                rec[p] = (A, B, v["doc_tok"][ok].astype(float), v["doc_n"][ok].astype(float))
            if len(rec) == 2: P[(g, a, b)] = rec
    return P


def pooled(rec, ix=None, ic=None):
    (Ax, Bx, _, _), (Ac, Bc, _, _) = rec["x"], rec["c"]
    if ix is not None: Ax, Bx, Ac, Bc = Ax[ix], Bx[ix], Ac[ic], Bc[ic]
    return math.log((Bx.sum() + Bc.sum()) / (Ax.sum() + Ac.sum())), math.log(Bx.sum() / Ax.sum()), math.log(Bc.sum() / Ac.sum())


def frac(P, fr=(0.125, 0.25, 0.5, 0.75, 1.0), nd=300):
    out = []
    rng = np.random.default_rng(3)
    for (g, a, b), rec in P.items():
        lr = pooled(rec)[0]
        if not 0.005 <= abs(lr) <= 0.15: continue
        Nx, Nc = len(rec["x"][0]), len(rec["c"][0]); row = {"golden": g, "a": a, "b": b, "logr": lr, "Nx": Nx, "Nc": Nc, "pts": []}
        for f in fr:
            nx, nc = max(2, round(f * Nx)), max(2, round(f * Nc)); v = []
            for _ in range(nd):
                if f < 1: ix, ic = rng.choice(Nx, nx, replace=False), rng.choice(Nc, nc, replace=False)
                else: ix, ic = rng.integers(0, Nx, Nx), rng.integers(0, Nc, Nc)     # f = 1: the bootstrap SE
                v.append(pooled(rec, ix, ic))
            v = np.array(v); row["pts"].append({"f": f, "sd_pool": float(v[:, 0].std()), "sd_x": float(v[:, 1].std()), "sd_c": float(v[:, 2].std()),
                                                "tok": float(rec["x"][2][:nx].sum() * Nx / max(nx, 1) * f + rec["c"][2].sum() * f)})
        full = row["pts"][-1]["sd_pool"]
        for p in row["pts"][:-1]:
            fpc = math.sqrt((1 - p["f"]))           # without replacement from the slice itself: SE^2 (1 - f) / f
            p["pred"] = full * math.sqrt(1 / p["f"]) * fpc
        out.append(row)
    return out


def simulate(d, X, C, w, rule, M=4000, seed=11):
    """P(accept), expected extra cost units for true effects d (array) with full-slice part SEs X (x), C (c) and the x
    part's KL share w (pooled = w d_x + (1 - w) d_c). rule = ("fixed", f) or ("seq", (F1, F2, ..., 1), (z1, z2, ...)):
    stage k has scored the cumulative fraction F_k of each part's documents (fresh documents each stage); it accepts
    iff the pooled z <= -z_k and both parts are < 0, rejects iff z >= z_k, else goes on; the last stage accepts iff the
    pooled and both parts are < 0. z_k = (za, zr) sets the accept and reject bounds apart (za = INF: reject early only).
    X, C: scalars or arrays like d (noise that grows with the effect). Returns P(accept), E[number of extra stages
    run], E[fraction scored] and a dict of per-d means: "rest" (the slice fraction an early accept leaves unscored: the
    new incumbent must still be scored on it), "early" (P(accept before the last stage)), "opt" (E[(pooled estimate -
    d) * accept]: the selection optimism banked with the accepts) and "opt_c" (the same for the calib part alone, in
    units of its full-slice SE C)."""
    rng = np.random.default_rng(seed); d = np.asarray(d, float)[:, None]; n = len(d)
    X = np.broadcast_to(np.asarray(X, float).reshape(-1, 1), (n, 1)); C = np.broadcast_to(np.asarray(C, float).reshape(-1, 1), (n, 1))
    if rule[0] == "fixed": F, Z = (rule[1],), ()
    else: F, Z = rule[1], rule[2]
    sx = sc = None; fprev = 0.0
    acc = np.zeros((n, M), bool); live = np.ones((n, M), bool); runs = np.zeros(n); scored = np.zeros(n)
    rest = np.zeros((n, M)); early = np.zeros((n, M), bool); opt = np.zeros((n, M)); opt_c = np.zeros((n, M))
    for k, f in enumerate(F):
        df = f - fprev
        ix = d * df + X * math.sqrt(df) * rng.standard_normal((n, M)); ic = d * df + C * math.sqrt(df) * rng.standard_normal((n, M))
        sx = ix if sx is None else sx + ix; sc = ic if sc is None else sc + ic          # sums: estimate = sum / f
        if k > 0: runs += live.mean(1)
        scored += live.mean(1) * df
        ex, ec = sx / f, sc / f; p = w * ex + (1 - w) * ec
        if k < len(Z):
            se = np.sqrt(w * w * X * X + (1 - w) ** 2 * C * C) / math.sqrt(f); z = p / se
            za, zr = Z[k] if isinstance(Z[k], tuple) else (Z[k], Z[k])
            a = live & (z <= -za) & (ex < 0) & (ec < 0); r = live & (z >= zr)
            rest[a] = 1 - f; early |= a
        else:
            a = live & (p < 0) & (ex < 0) & (ec < 0); r = live
        opt += np.where(a, p - d, 0); opt_c += np.where(a, (ec - d) / C, 0)
        acc |= a; live &= ~(a | r)
        fprev = f
    return acc.mean(1), runs, scored, {"rest": rest.mean(1), "early": early.mean(1), "opt": opt.mean(1), "opt_c": opt_c.mean(1)}


def rname(rule):
    zs = lambda z: (f"a{z[0]:g}".replace("ainf", "a-") + f"r{z[1]:g}") if isinstance(z, tuple) else f"{z:g}"
    return f"fixed {rule[1]:g}" if rule[0] == "fixed" else "seq " + ">".join(f"{f:g}" for f in rule[1]) + " z" + "/".join(zs(z) for z in rule[2])


REC = None   # rname(REC_RULE), set below


def v2_moves(path):
    """V2's search history (data/v2/results.jsonl): every knapsack candidate (mean, Thompson, local search, tilted /
    random) against the best mix measured before it at the same slice, target and position count, |delta bpw| < 0.02:
    log(KL / best-so-far KL). Effects include V2's slice noise (paired SE ~1.7-1.9%), so the spread is a slight
    overestimate. Returns {"V2 all": array, "V2 ls": array} (empty without a path)."""
    if not path: return {}
    best, d, ls = {}, [], []
    for l in open(path):
        r = json.loads(l)
        m = re.match(r"^(a|f2|f3|w)-knap-(?:r\d+|s\d+)(?:oo)?(?:-(ts|ls|tl|rd)\d)?-sc4\.00-e4-([\d.]+)$", r["tag"])
        if not m or r.get("kld") is None: continue
        k = (m.group(1), m.group(3), r["positions"]); bpw = r["bytes"] * 8 / 26895998464
        if k in best and abs(bpw - best[k][1]) < 0.02:
            d.append(math.log(r["kld"] / best[k][0]))
            if m.group(2) == "ls": ls.append(d[-1])
        if k not in best or r["kld"] < best[k][0]: best[k] = (r["kld"], bpw)
    return {"V2 all": np.array(d), "V2 ls": np.array(ls)}


def decide(SE, w, cost, priors, rules, svals=(0.01, 0.02, 0.04, 0.08)):
    """SE = {"x", "c"}: full-slice part SEs per unit of calib per-document noise; cost(extra_runs, frac, rest, early)
    -> seconds. priors: (mu, sigma) normal, or ("emp", NAME, samples, (s0, q)): an empirical distribution of move
    effects (200 quantiles) whose per-document noise grows with the effect, s = sqrt(s0^2 + (q d)^2) (svals unused)"""
    out = []
    for pr in priors:
        if pr[0] == "emp":
            d = np.quantile(pr[2], (np.arange(200) + 0.5) / 200); pd = np.full(len(d), 1 / len(d))
            s0, q = pr[3]; S_ = [(f"{s0:g}+{q:g}|d|", np.sqrt(s0 * s0 + (q * d) ** 2))]; mu, sig = pr[1], None
        else:
            mu, sig = pr; d = np.linspace(mu - 5 * sig, mu + 5 * sig, 161); pd = norm.pdf(d, mu, sig); pd /= pd.sum()
            S_ = [(s, s) for s in svals]
        oracle = float(np.sum(pd * np.maximum(-d, 0)))
        for sn, s in S_:                                   # per-document noise (calib units) of the move
            X, C = SE["x"] * s, SE["c"] * s
            for rule in rules:
                pa, pe, fs, ex = simulate(d, X, C, w, rule)
                E = lambda v: float(np.sum(pd * v))
                c = cost(E(pe), E(fs), E(ex["rest"]), E(ex["early"])); h = 3600 / c
                gain = E(pa * -d); corr = E(np.where(d < 0, pa, 1 - pa)); wrong = E(np.where(d > 0, pa, 0))
                out.append({"s": sn, "mu": mu, "sigma": sig, "rule": rname(rule), "cost_s": round(c, 1),
                            "extra_runs": round(E(pe), 3), "frac_scored": round(E(fs), 3), "rest": round(E(ex["rest"]), 3),
                            "p_accept": round(E(pa), 3), "gain_frac": round(gain / oracle, 3), "correct": round(corr, 3),
                            "wrong_accept": round(wrong, 4), "gain_per_h": round(100 * gain * h, 3), "correct_per_h": round(corr * h, 2),
                            "evals_per_h": round(h, 1),
                            # selection optimism: the share of the accepts' measured gain that is noise, and the calib
                            # part's noise per accept in units of its full-slice SE (rotation, section 5)
                            "opt_share": round(-E(ex["opt"]) / max(gain, 1e-12), 3),
                            "opt_c_per_acc": round(E(ex["opt_c"]) / max(E(pa), 1e-12), 3)})
    return out


REC = rname(REC_RULE)


def resample(P, rules, targets=(-0.01, -0.005, 0.005, 0.01), nd=400):
    """real documents: each 27d-f pair with 1% <= |log r| <= 15% has its per-document log ratios scaled to the target
    delta (pooled), slices of the pair's own sizes are drawn with replacement (stage k = the first F_k share of the draw)
    and the rule applied with the delta-method SE of the drawn documents. Beside it the normal model's P(accept) for the
    same pair (its own part SEs scaled the same way, its own KL share, delta = the scaled pooled effect): the check that
    the model's decisions hold on heavy-tailed real documents."""
    rng = np.random.default_rng(7); out = []
    for (g, a, b), rec in P.items():
        lr = pooled(rec)[0]
        if not 0.01 <= abs(lr) <= 0.15: continue
        Nx, Nc = len(rec["x"][0]), len(rec["c"][0])
        for tg in targets:
            k = tg / lr
            R = {p: (A, np.where(A > 0, A * (B / np.where(A > 0, A, 1)) ** k, B)) for p, (A, B, _, _) in rec.items()}
            Ax, Bx = R["x"]; Ac, Bc = R["c"]
            true = math.log((Bx.sum() + Bc.sum()) / (Ax.sum() + Ac.sum()))
            w = Ax.sum() / (Ax.sum() + Ac.sum())
            X, C = boot_se(Ax, Bx, 400), boot_se(Ac, Bc, 400)
            ix_all, ic_all = rng.integers(0, Nx, (nd, Nx)), rng.integers(0, Nc, (nd, Nc))
            for rule in rules:
                F, Z = ((rule[1],), ()) if rule[0] == "fixed" else (rule[1], rule[2])
                acc = 0
                for t in range(nd):
                    for kk, f in enumerate(F):
                        nx, nc = max(2, round(f * Nx)), max(2, round(f * Nc)); ix, ic = ix_all[t, :nx], ic_all[t, :nc]
                        a_ = np.concatenate([Ax[ix], Ac[ic]]); b_ = np.concatenate([Bx[ix], Bc[ic]])
                        d = math.log(b_.sum() / a_.sum()); dx = math.log(Bx[ix].sum() / Ax[ix].sum()); dc = math.log(Bc[ic].sum() / Ac[ic].sum())
                        if kk < len(Z):
                            u = b_ / b_.mean() - a_ / a_.mean(); se = u.std(ddof=1) / math.sqrt(len(u))
                            za, zr = Z[kk] if isinstance(Z[kk], tuple) else (Z[kk], Z[kk])
                            if d / se <= -za and dx < 0 and dc < 0: acc += 1; break
                            if d / se >= zr: break
                        else:
                            acc += d < 0 and dx < 0 and dc < 0
                pm = float(simulate([true], X, C, w, rule, M=4000)[0][0])
                out.append({"golden": g, "a": a, "b": b, "logr": lr, "target": tg, "true": true, "w": w, "X": X, "C": C,
                            "rule": rname(rule), "p_accept": acc / nd, "p_model": pm})
    return out


def main():
    npz, timing, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
    G = S.load(npz); IX = json.load(open(npz[:-4] + ".index.json")); T = json.load(open(timing))
    for kv in sys.argv[4:]:                                   # key=value overrides of the cost model (e.g. runs_per_eval=1)
        k, v = kv.split("=", 1); T[k] = v if k == "v2" else float(v)
    skip4 = T.pop("skip_resample", 0)
    # 1 noise
    N = noise(G, IX); json.dump(N, open(f"{od}/noise.json", "w"))
    print("== 1 per-document noise s = SE*sqrt(docs) and q = s/|log r| (pairs 0.5% <= |log r| <= 15%;"
          " move1 = a 3.00 -> 3.50 group move vs V2 3.00, move2 = two disjoint group moves (a swap-like pair), c / x = other pairs)")
    for k in ("move1", "move2", "c", "x"):
        R = [r for r in N if r["kind"] == k]
        if not R: continue
        s = np.array([r["s"] for r in R]); q = np.array([r["q"] for r in R]); a = np.array([abs(r["logr"]) for r in R])
        print(f"  {k:5s} pairs {len(R):4d}  |log r| median {100*np.median(a):.2f}%  s median {np.median(s):.3f} [IQR {np.percentile(s,25):.3f}-{np.percentile(s,75):.3f}]"
              f"  q median {np.median(q):.2f} [IQR {np.percentile(q,25):.2f}-{np.percentile(q,75):.2f}]")
    # x vs c on the same pair (two-part slices)
    P = two_part(G)
    rx, rows = [], []
    for (g, a, b), rec in P.items():
        lr = pooled(rec)[0]
        if not 0.005 <= abs(lr) <= 0.15: continue
        sx = boot_se(*rec["x"][:2]) * math.sqrt(len(rec["x"][0])); sc = boot_se(*rec["c"][:2]) * math.sqrt(len(rec["c"][0]))
        rx.append(sx / sc); rows.append((g, a, b, lr, sx, sc))
    global RX
    RX = float(np.median(rx)) if rx else 1.0
    print(f"  x / c per-document noise on the same pair (27d-f, {len(rx)} pairs): median {RX:.2f}  IQR {np.percentile(rx,25):.2f}-{np.percentile(rx,75):.2f}")
    # 2 frac
    F = frac(P); json.dump(F, open(f"{od}/frac.json", "w"))
    print("== 2 pooled SE when each part keeps a fraction f of its documents (27d-f pairs), sd / predicted (1/sqrt f)")
    for f in (0.125, 0.25, 0.5, 0.75):
        r = [p["sd_pool"] / p["pred"] for row in F for p in row["pts"] if p["f"] == f and p.get("pred")]
        print(f"  f {f:5.3f}: ratio median {np.median(r):.2f}  IQR {np.percentile(r,25):.2f}-{np.percentile(r,75):.2f}  (n {len(r)})")
    # the standard slice (mkslice27 defaults): T["x_docs"] xcorp pieces + T["calib_docs"] calib documents; the x part's
    # KL share w and tokens per piece from the 27d-f pairs (A side), calib tokens = T["calib_tok"]
    W = [rec["x"][0].sum() / (rec["x"][0].sum() + rec["c"][0].sum()) for rec in P.values()]
    w = float(np.median(W)); tok_x = float(np.median([rec["x"][2].mean() for rec in P.values()]))
    slice_tok = T["x_docs"] * min(tok_x, T.get("x_tok_cap", 1e9)) + T["calib_tok"]
    SE = {"x": RX / math.sqrt(T["x_docs"]), "c": 1 / math.sqrt(T["calib_docs"])}
    tau = T["s_per_tok"]
    def cost(extra_runs, frac, rest=0.0, early=0.0, runs=T["runs_per_eval"]):
        """one candidate: radmix + a radiance run per part (goldcat: 1) per stage + the tokens scored; an early accept
        (before the last stage) also scores the new incumbent on the rest of the slice (one more run); plus the
        incumbent's full-slice eval once per T["decisions_per_slice"] decisions (the slice's rotation)"""
        c = T["mix_s"] + runs * T["run_s"] * (1 + extra_runs + early) + tau * slice_tok * (frac + rest)
        return c + (T["mix_s"] + runs * T["run_s"] + tau * slice_tok) / T["decisions_per_slice"]
    print(f"  standard slice: {T['x_docs']} x pieces ({tok_x:.0f} tokens each in 27d-f) + {T['calib_docs']} calib docs ({T['calib_tok']} tokens):"
          f" {slice_tok/1000:.0f}k tokens, x KL share w {w:.2f}; SE per unit calib per-doc noise: x {SE['x']:.4f}, c {SE['c']:.4f},"
          f" pooled {math.sqrt(w*w*SE['x']**2+(1-w)**2*SE['c']**2):.4f}; full eval {cost(0, 1.0):.0f} s")
    # 3 decide: normal priors (close calls, the task's ~0.5-2% moves) and V2's measured candidate effects
    rules = RULES
    priors = [(0.005, 0.01), (0.0, 0.01), (0.005, 0.005), (0.01, 0.02)]
    V2 = v2_moves(T.pop("v2", None))
    for name, smp in V2.items(): priors.append(("emp", name, smp, (0.01, 2.0)))
    D = decide(SE, w, cost, priors, rules)
    json.dump({"SE": SE, "w": w, "RX": RX, "slice_tok": slice_tok, "timing": T, "rows": D}, open(f"{od}/decide.json", "w"), indent=1)
    print("== 3 decisions (normal model, same sign on both parts): per card-hour, prior delta ~ N(mu, sigma) or V2's measured"
          " candidates (emp, s = 0.01 + 2|d|), per-doc noise s")
    print(f"  {'s':>9s} {'mu':>6s} {'sig':>5s} {'rule':30s} {'cost_s':>6s} {'frac':>5s} {'rest':>5s} {'P(acc)':>6s} {'correct':>7s} {'wrongacc':>8s} {'gain%':>6s}"
          f" {'gain%/h':>7s} {'corr/h':>6s} {'opt':>5s} {'optc':>5s}")
    for r in D:
        mu = f"{100*r['mu']:5.1f}%" if r["sigma"] is not None else f"{r['mu']:>6s}"; sg = f"{100*r['sigma']:4.1f}%" if r["sigma"] is not None else "  emp"
        sn = f"{r['s']:9.3f}" if r["sigma"] is not None else f"{r['s']:>9s}"
        print(f"  {sn} {mu} {sg} {r['rule']:30s} {r['cost_s']:6.1f} {r['frac_scored']:5.2f} {r['rest']:5.2f} {r['p_accept']:6.3f} {r['correct']:7.3f} {r['wrong_accept']:8.4f}"
              f" {r['gain_frac']:6.3f} {r['gain_per_h']:7.3f} {r['correct_per_h']:6.1f} {r['opt_share']:5.2f} {r['opt_c_per_acc']:5.2f}")
    # summary: gain per card-hour relative to fixed 1 (normal priors: s <= 0.04, the moves measured here -- single group
    # moves s 0.04 at |log r| 4.5%, swap-like pairs 0.07 at 2.2%; one-tensor moves are smaller -- and the V2 priors)
    key = lambda r: (r["s"], r["mu"], r["sigma"])
    base = {key(r): r for r in D if r["rule"] == "fixed 1"}
    for title, sel in (("normal priors, s in {0.01, 0.02, 0.04} x 4 priors", lambda r: r["sigma"] is not None and r["s"] <= 0.04),
                       ("normal priors, s = 0.08 (noisy moves)", lambda r: r["sigma"] is not None and r["s"] == 0.08),
                       ("V2's measured candidates (all kinds; local search only)", lambda r: r["sigma"] is None)):
        print(f"== 3b {title}: gain/h and correct/h relative to fixed 1")
        for rn in dict.fromkeys(r["rule"] for r in D):
            v = [(r["gain_per_h"] / base[key(r)]["gain_per_h"], r["correct_per_h"] / base[key(r)]["correct_per_h"],
                  r["wrong_accept"] / max(base[key(r)]["wrong_accept"], 1e-9), r["cost_s"]) for r in D if r["rule"] == rn and sel(r)]
            if not v: continue
            g = np.array([x[0] for x in v]); c = np.array([x[1] for x in v]); wa = np.array([x[2] for x in v]); cs = np.array([x[3] for x in v])
            print(f"  {rn:32s} gain/h x{np.median(g):5.2f} (min {g.min():+5.2f})  correct/h x{np.median(c):4.2f}  wrong accepts x{np.median(wa):4.2f}"
                  f" (max {wa.max():4.2f})  s/decision {np.median(cs):5.0f}")
    # 5 rotation: the calib part's selection optimism. An accept banks E[e_c | accept] (units of that move's calib SE);
    # moves have independent per-document noise (w123b2: cross-move corr ~0), so after K accepts on the same calib
    # documents the incumbent's calib-part level is optimistic by ~ opt_c sqrt(K) SEs of its cumulative change: K at 1 SE
    # = 1 / opt_c^2. Decisions per rotation = K / P(accept).
    print("== 5 rotation: calib-part optimism per accept (opt_c, in SE_c units) and decisions until the cumulative optimism reaches 1 / 2 SE")
    for r in D:
        if r["rule"] not in ("fixed 1", "fixed 0.25", REC) or (r["sigma"] is not None and r["s"] not in (0.02, 0.04)): continue
        oc = abs(r["opt_c_per_acc"]); k1, k2 = 1 / oc ** 2, 4 / oc ** 2
        print(f"  {str(r['s']):>9s} {str(r['mu']):>6s} {str(r['sigma']):>6s} {r['rule']:32s} P(acc) {r['p_accept']:.3f}  opt share {r['opt_share']:5.2f}"
              f"  opt_c {r['opt_c_per_acc']:+.2f}  accepts to 1 SE {k1:5.1f} / 2 SE {k2:5.1f}  decisions {k1/max(r['p_accept'],1e-9):6.0f} / {k2/max(r['p_accept'],1e-9):6.0f}")
    if skip4: return
    # 4 resample
    Rr = resample(P, [("fixed", 1.0), ("fixed", 0.25), ("seq", (0.25, 1.0), (2.0,)), ("seq", (0.125, 0.5, 1.0), (1.5, 1.5)), REC_RULE])
    json.dump(Rr, open(f"{od}/resample.json", "w"))
    print("== 4 real documents (27d-f pairs scaled to delta, their own slice sizes): P(accept), real vs normal model")
    for rule in dict.fromkeys(r["rule"] for r in Rr):
        for tg in sorted({r["target"] for r in Rr}):
            v = [r for r in Rr if r["rule"] == rule and r["target"] == tg]
            pr = np.array([r["p_accept"] for r in v]); pm = np.array([r["p_model"] for r in v])
            print(f"  {rule:32s} delta {100*tg:+5.1f}%  P(accept) real mean {pr.mean():.3f} [p10 {np.percentile(pr,10):.3f}]  model mean {pm.mean():.3f}"
                  f"  mean |real - model| {np.abs(pr-pm).mean():.3f}  n {len(v)}")


if __name__ == "__main__":
    main()
