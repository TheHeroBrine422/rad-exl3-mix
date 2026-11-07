#!/usr/bin/env python3
"""mkjobs.py OUTDIR [COSTFILE [ARMNAME]] -- #46-03 step 3 knap5a jobs (run6.py format: name, env, targets, lambdas), w63's corrected
swj curve and V3R menu (v3/levels/reprice/mkjobs2.py 'main', fast variant), a finer lambda grid for the KLD-time curves:
  noa4   A8 only (KN_NOA4=1): the reference fast variant
  a4e1c  + A4 g64 / g128 on the i4 levels, E1c proxy costs (w58's 'a4' arm, re-run with the corrected swj)
  a4m    (ARMNAME) + A4 at the measured 27B costs (KN_A4COST=COSTFILE, mkcost.py); also at target - 0.03 / - 0.06 (step 4's bytes test)
A4 times: a4g64 = a8 / 1.31, a4g128 = a8 / 1.51 (w34's K2 layer ratios)."""
import sys

LAMS = ["1e-08", "2e-08", "3e-08", "5e-08", "1e-07", "2e-07", "3e-07", "5e-07", "1e-06", "2e-06", "3e-06", "5e-06", "1e-05", "3e-05", "1e-04", "3e-04"]
NEW = ("KN_V3R=1 KN_EMBT=2 KN_HEADSLAB=1 KN_LEVELS=../../data/v3/levels KN_DAMP=1 KN_TP=../../twopass-m2048.jsonl KN_PICKS=1 "
       "KN_SWJ=all KN_SWJ_PTS=2:0.91,2.5:0.935,4:0.989,5:1.0")
out, cost = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
name = sys.argv[3] if len(sys.argv) > 3 else "a4m"     # one name per cost table version (run6 skips finished names)
ARMS = [("noa4", f"{NEW} KN_NOA4=1", [2.0, 2.5, 3.0]), ("a4e1c", f"{NEW} KN_A4X128=1.51", [2.0, 2.5, 3.0])]
if cost: ARMS.append((name, f"{NEW} KN_A4X128=1.51 KN_A4COST={cost}", [2.0, 1.97, 1.94, 2.5, 2.47, 2.44, 3.0, 2.97, 2.94]))
js = [f"{n}-f-{s:g}-l{lm}\t{env}\t{s:g}\t{lm}" for n, env, ss in ARMS for s in sorted(ss, reverse=True) for lm in LAMS]
open(out, "w").write("\n".join(js) + "\n"); print(len(js), "jobs")
