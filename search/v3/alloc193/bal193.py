# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""bal193.py -- RULES-193 §3 / §5 test construction: force options on a set of weights, then bring the bytes into the
size's window with my fit.py refit's cheapest moves OUTSIDE the set (downgrade the smallest predicted loss per byte
while above the window top; then upgrade the best predicted gain per byte that fits)."""
import numpy as np

import common193 as C


def shift(M, a, w, steps):
    """move weight w by `steps` options of its own family (clamped)"""
    opts = sorted([o for o in M.T[M.WI[w]]["opts"] if C.KM.fam(o["o"]) == C.KM.fam(a[w])], key=lambda o: o["bytes"])
    names = [o["o"] for o in opts]; j = names.index(a[w]); jj = min(max(j + steps, 0), len(opts) - 1)
    return names[jj]


def balance(pred, M, a, fixed_set, s0, mode="decide"):
    lo, hi = C.window(M, s0)
    if mode == "probe": lo, hi = (s0 - 0.02) * M.P / 8, (s0 + 0.0) * M.P / 8
    D = pred.D; W = D.W; free = [i for i, w in enumerate(W) if w not in fixed_set]
    fam = {w: C.KM.fam(a[w]) for w in W}
    ok = [np.array([C.KM.fam(o["o"]) == fam[w] for o in M.T[M.WI[w]]["opts"]]) for w in W]
    b = M.bytes(a)
    for _ in range(5000):
        if b <= hi: break
        J, dl = pred.lin(a); best = None
        for i in free:
            j = J[i]
            for jj in range(j - 1, -1, -1):
                if ok[i][jj]: break
            else: continue
            db = pred.B[i][jj] - pred.B[i][j]
            if db >= 0: continue
            v = dl[i][jj] / -db
            if best is None or v < best[0]: best = (v, i, jj, db)
        if best is None: return None
        a = pred.assign([best[2] if k == best[1] else j for k, j in enumerate(J)], a); b += best[3]
    for _ in range(5000):
        J, dl = pred.lin(a); best = None
        for i in free:
            j = J[i]
            for jj in range(j + 1, len(pred.B[i])):
                if ok[i][jj]: break
            else: continue
            db = pred.B[i][jj] - pred.B[i][j]
            if db <= 0 or b + db > hi or dl[i][jj] >= 0: continue
            v = dl[i][jj] / db
            if best is None or v < best[0]: best = (v, i, jj, db)
        if best is None: break
        a = pred.assign([best[2] if k == best[1] else j for k, j in enumerate(J)], a); b += best[3]
    return a if lo <= M.bytes(a) <= hi else None
