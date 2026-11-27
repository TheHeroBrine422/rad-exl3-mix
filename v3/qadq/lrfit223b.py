# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""lrfit223b.py LRTAB.json -- #223: the pooled form (c + b1 x + b2 x^2 + b3 bpw + b4 x bpw, x = log2 lr) fitted on overlapping size
windows (2.0-2.5, 2.5-3.5, 3.0-4.0), flat arms only; per response the optimum per size in the window (+ 10-90 % bootstrap over
residuals, 2,000 draws), the predicted ratio there, rmse and points / parameters."""
import json, sys
import numpy as np
R = [r for r in json.load(open(sys.argv[1])) if r["sched"].startswith("flat") and r.get("JC")]
WIN = [(2.0, 2.5), (2.5, 3.0, 3.5), (3.0, 3.5, 4.0)]
for key, sub in (("JC", "med"), ("JC", "r"), ("JC", "wo2"), ("JG", "med"), ("J217", "med")):
    print(f"== {key}/{sub}")
    for win in WIN:
        P = [(r["size"], np.log2(r["lr"]), np.log(r[key][sub])) for r in R if r["size"] in win and r.get(key)]
        S, X, Y = map(np.array, zip(*P)); A = np.c_[np.ones_like(X), X, X ** 2, S, X * S]
        b, *_ = np.linalg.lstsq(A, Y, rcond=None); res = Y - A @ b; rng = np.random.default_rng(0)
        bs = [np.linalg.lstsq(A, A @ b + rng.choice(res, len(res)), rcond=None)[0] for _ in range(2000)]
        opt = lambda bb, s: -(bb[1] + bb[4] * s) / (2 * bb[2]) if bb[2] > 0 else np.nan
        out = []
        for s in win:
            o = opt(b, s)
            if not np.isfinite(o): out.append(f"{s}: no minimum (b2 {b[2]:+.4f})"); continue
            ob = np.array([opt(bb, s) for bb in bs]); ob = ob[np.isfinite(ob)]
            pr = np.exp(b[0] + b[1] * o + b[2] * o * o + b[3] * s + b[4] * o * s)
            out.append(f"{s}: {2 ** o:.1e} [{2 ** np.percentile(ob, 10):.1e}, {2 ** np.percentile(ob, 90):.1e}] pred {pr:.3f}")
        print(f"  {'-'.join(map(str, win)):12s} n {len(Y)} rmse {np.sqrt(np.mean(res ** 2)):.4f}  " + "  ".join(out))
