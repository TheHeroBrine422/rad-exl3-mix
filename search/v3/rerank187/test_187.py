# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_187.py -- checks for the #187 replay (run: python3 test_187.py SNAP [FIT0000_IN]).
1. add_obs reproduction: the rows the replay builds after result 6 equal the driver's fit input fits/fit0000-all.in.json.
2. pool187 with all deviations 0 == fit.py's SModel (predict_rows and residuals on the same rows).
3. pool187 band tables (tables_band) == its row predictions (predict_rows / exp(soff)), with non-zero deviations.
4. the candidate table: whole-slice / audit / band counts as RULES-187 section 1 describes."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay as R          # noqa: E402
import fit as FT            # noqa: E402
import pool187 as PL        # noqa: E402


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
    if not ok: check.failed += 1
check.failed = 0


def main(argv):
    snap = argv[0]
    S = R.Snap(snap)
    fit0 = argv[1] if len(argv) > 1 else os.path.join(snap, "fit0000-all.in.json")   # copy of the driver's fit input
    if os.path.exists(fit0):
        fi = json.load(open(fit0)); pri = {o["tag"] for o in S.prior}
        drv = {(o["tag"], o["label"]): o for o in fi["obs"] if o["tag"] not in pri}
        mine = {(o["tag"], o["label"]): o for o in S.obs_at(6)}
        same = set(drv) == set(mine) and all(abs(drv[k]["kld"] - mine[k]["kld"]) < 1e-12 and abs(drv[k]["se"] - mine[k]["se"]) < 1e-12
                                             and drv[k]["assign"] == mine[k]["assign"] and drv[k].get("size") == mine[k].get("size") for k in drv)
        check("add_obs reproduces the driver's fit0000 rows", same, f"{len(drv)} rows")
    else:
        check("add_obs reproduces the driver's fit0000 rows", False, "fit0000 input not found (skipped)")

    obs = S.obs_at(40) + S.prior
    opts = S.conf.get("fit_opts", {})
    base = FT.build(S.M, obs, opts); xb = base.x0()
    pool = PL.build_pool(S.M, obs, opts, 0.3)
    pool.layout(); xp = np.zeros(pool.N); xp[:len(xb)] = xb
    idx = list(range(len(base.D.rows)))
    pb = base.predict_rows(xb, idx); pp = pool.predict_rows(xp, idx)
    rb = base.residuals(xb, idx); rp = pool.residuals(xp, idx)
    check("pool(0 deviations) == fit.py predict_rows", np.allclose(pb, pp, rtol=1e-10, atol=0), f"max|d| {np.max(np.abs(pb - pp)):.2e}")
    check("pool(0 deviations) == fit.py residuals (data + shared priors)", np.allclose(rb, rp[:len(rb)], rtol=1e-10, atol=0))

    rng = np.random.default_rng(187)
    xr = xp.copy(); L = pool.layout()
    xr[L["dth"]] = rng.normal(0, 0.1, L["dth"].stop - L["dth"].start); xr[L["dq"]] = rng.normal(0, 0.1, 4)
    xr[L["dka"]] = rng.normal(0, 0.1, L["dka"].stop - L["dka"].start)
    rows = list(range(0, len(pool.D.rows), 3))
    pr = pool.predict_rows(xr, rows)
    # xscale stays on (toggling it changes layout(): lamx's slot moves the deviation indices)
    tab = []
    for i, p in zip(rows, pr):
        k = int(pool.band[i]); tb = pool.tables_band(xr, k)
        pt = pool.predict(xr, obs[i]["assign"], tb)                    # table path: P without soff, no lambda
        so = math.exp(pool.unpack(xr)["soff"][pool.SLI[i]]); lam = pool.lam(xr)
        want = so * pt ** lam if pool.XM[i] else so * pt
        tab.append(abs(math.log(want) - math.log(p)))
    check("band tables == band row predictions (non-zero deviations, xscale as fitted)", max(tab) < 1e-8,
          f"max|dlog| {max(tab):.2e} over {len(tab)} rows")

    cands = S.cands
    whole = [c for c in cands if not c["stopped"]]
    ok = len(whole) == 107 and sum(c["audit"] for c in whole) == 27 and sum(c["audit"] for c in cands) == 27
    check("candidate table: 258 candidates, 107 whole-slice, 27 audit", len(cands) == 258 and ok, f"{len(cands)} / {len(whole)} / {sum(c['audit'] for c in cands)}")
    print(f"{'FAILED ' + str(check.failed) if check.failed else 'all checks passed'}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
