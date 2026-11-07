#!/usr/bin/env python3
"""mkjobs.py OUTDIR -- w58 job lists for run5.py: jobs-cpu2.txt (Thor) and jobs-cpu1.txt (colo).
One job = one arm x size x lambda (accurate: KN_MODES=f16, lambda 0; fast: a8-g128 allowed, lambda grid)."""
import sys

def st(a, b, s): return ",".join(f"{a + i * s:g}" for i in range(int(round((b - a) / s)) + 1))

SIZES = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
HI = [5.0, 5.5, 6.0, 6.5, 7.0]
LAMS = ["1e-08", "3e-08", "1e-07", "3e-07", "1e-06", "3e-06", "1e-05"]
LAMS_S = LAMS[3:]   # sensitivity arms (trimmed 01:25 UTC, after the main arm's grid: the fast pick sits at >= 3e-7)
FULL = {"main", "old", "oldf", "pass1", "i4low", "v2menu"}   # arms on the full grid (oldf, pass1, i4low: kept as started)
OLD = "KN_V3=1 KN_FRAC=1 KN_EMBT=2 KN_HEADSLAB=1 KN_LEVELS=../../data/v3/levels KN_DAMP=1"
NEW = "KN_V3R=1 KN_EMBT=2 KN_HEADSLAB=1 KN_LEVELS=../../data/v3/levels KN_DAMP=1 KN_TP=../../twopass-m2048.jsonl KN_PICKS=1"
I8_DEF = st(1.125, 5, 0.125) + "," + st(5.25, 7, 0.25)
ARMS = [   # name, env, sizes, accurate?, fast?, box
    ("main", f"{NEW} KN_NOA4=1 KN_SWJ=all", SIZES, 1, 1, "cpu2"),
    ("old", f"{OLD} KN_SPEED=../bench/data/speedcal-stock.jsonl", SIZES, 1, 0, "cpu2"),
    ("oldf", f"{OLD} KN_TP=../../twopass-m2048.jsonl KN_GROUPS=128 KN_NOA4=1", SIZES, 0, 1, "cpu2"),
    ("cd", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_CD=0.99", SIZES, 1, 1, "cpu2"),
    ("noswj", f"{NEW} KN_NOA4=1", SIZES, 1, 1, "cpu2"),
    ("gu", f"{NEW} KN_NOA4=1 KN_SWJ=gu", SIZES, 1, 1, "cpu2"),
    ("h16", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_HEAD16=1", SIZES, 1, 0, "cpu1"),   # head: f16, no prefill time
    ("pass1", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_I4= KN_I8={st(1.25, 5, 0.25)},5.5,6,6.5,7 KN_M1=6,6.5,7,7.5,8", SIZES, 1, 1, "cpu1"),
    ("i4low", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_I8={st(2.125, 5, 0.125)},{st(5.25, 7, 0.25)}", SIZES[:7], 1, 1, "cpu1"),
    ("fine57", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_I8={st(1.125, 7, 0.125)} KN_M1={st(6, 8, 0.125)}", HI, 1, 1, "cpu1"),
    ("r8ratio", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_R8_MID=ratio", HI[1:], 0, 1, "cpu1"),
    ("phifloor", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_PHI_INTERP=0", HI, 1, 1, "cpu1"),
    ("a4", f"{NEW} KN_SWJ=all", SIZES[:5], 0, 1, "cpu1"),
    ("v2menu", "KN_EMBT=0", SIZES, 1, 0, "cpu1"),        # knap4's V2 menu (k4grid 'v2'): the ratio base for '-x% vs V2'
    # added 02:20 UTC after the first results:
    ("phifloor", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_PHI_INTERP=0", SIZES[:11], 1, 0, "cpu1"),
    # added 03:10 UTC (Thor, after its queue): value of the duplicate / fast-only top levels
    ("dupm1", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_M1={st(6.5, 8, 0.25)}", HI, 1, 0, "cpu2b"),
    ("noi8k7", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_I8={st(1.125, 5, 0.125)},{st(5.25, 6.75, 0.25)}", [6.5, 7.0], 0, 1, "cpu2b"),
    ("noi8hi", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_I8={st(1.125, 5, 0.125)},{st(5.25, 6, 0.25)}", [5.5, 6.0, 6.5, 7.0], 0, 1, "cpu2b"),
    ("nolow", f"{NEW} KN_NOA4=1 KN_SWJ=all KN_I8={st(1.375, 5, 0.125)},{st(5.25, 7, 0.25)}", SIZES[:2], 1, 0, "cpu1"),
]
out = {"cpu1": [], "cpu2": [], "cpu2b": []}
for name, env, sizes, acc, fast, box in ARMS:
    for s in sorted(sizes, reverse=True):          # big sizes are the slow solves: start them first
        if acc: out[box].append(f"{name}-acc-{s:g}\t{env} KN_MODES=f16\t{s:g}\t0")
        if fast:
            for lm in (LAMS if name in FULL else LAMS_S[1:] if box == "cpu2b" else LAMS_S): out[box].append(f"{name}-f-{s:g}-l{lm}\t{env}\t{s:g}\t{lm}")
for box, js in out.items():
    open(f"{sys.argv[1]}/jobs-{box}.txt", "w").write("\n".join(js) + "\n")
    print(box, len(js))
