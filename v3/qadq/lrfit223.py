# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""lrfit223.py LRTAB.json -- #223 pooled lr fit over the flat arms: y = log(ratio) = c + b1 x + b2 x^2 + b3 bpw + b4 x bpw, x = log2 lr;
per-size optimum x* = -(b1 + b4 bpw) / (2 b2). Responses: jc228 median doc, jc228 w/o 2 worst docs, j217 median doc, gap median doc.
CI of the optimum: bootstrap over arms (resample residuals, 2,000 draws)."""
import json, sys
import numpy as np
R = [r for r in json.load(open(sys.argv[1])) if r["sched"].startswith("flat")]
def fit(key, sub):
    pts = [(r["size"], np.log2(r["lr"]), np.log(r[key][sub])) for r in R if r.get(key) and sub in r[key]]
    S, X, Y = map(np.array, zip(*pts)); A = np.c_[np.ones_like(X), X, X**2, S, X*S]
    b, *_ = np.linalg.lstsq(A, Y, rcond=None); res = Y - A @ b
    opt = lambda b, s: -(b[1] + b[4] * s) / (2 * b[2]) if b[2] > 0 else np.nan
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        Yb = A @ b + rng.choice(res, len(res)); bb, *_ = np.linalg.lstsq(A, Yb, rcond=None); bs.append(bb)
    out = {"n": len(Y), "rmse": float(np.sqrt(np.mean(res**2))), "b": b.tolist(), "opt": {}}
    for s in (2.0, 2.5, 3.0, 3.5, 4.0):
        o = np.array([opt(bb, s) for bb in bs]); o = o[np.isfinite(o)]
        out["opt"][s] = (2 ** opt(b, s), 2 ** np.percentile(o, 10), 2 ** np.percentile(o, 90), float(np.exp(b[0] + b[1]*opt(b,s) + b[2]*opt(b,s)**2 + b[3]*s + b[4]*opt(b,s)*s)))
    return out
for key, sub in (("JC", "med"), ("JC", "wo2"), ("J217", "med"), ("JG", "med"), ("JG", "wo2")):
    o = fit(key, sub)
    print(f"{key}/{sub}: n {o['n']} rmse {o['rmse']:.4f} b2 {o['b'][2]:+.4f}  " + "  ".join(f"{s}: {v[0]:.2e} [{v[1]:.1e}, {v[2]:.1e}] pred {v[3]:.3f}" for s, v in o["opt"].items()))
