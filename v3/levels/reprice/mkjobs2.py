#!/usr/bin/env python3
"""mkjobs2.py OUTDIR -- w63 rerun of w58's swj arms with the corrected swj curve (LOG 2026-11-06 7:50 PM CST: the K4
x0.78 was a head mismatch). Same solver (knap5.py), env, sizes and lambda grids as mkjobs.py; only KN_SWJ_PTS changes:
whole-model KL ratio r(K) = K2 0.91 (0.8B shape, -8.6%), K2.5 0.935 (27B verify), K4 0.989 (27B head-matched), K5 1.0;
knap5's lin(): linear in K between the points, flat outside (<= K2: 0.91, >= K5: 1.0).
Arms: main (KN_SWJ=all), gu (KN_SWJ=gu, gate_up factor re-solved at each point), noi8k7 (fast 6.5 / 7.0 without i8 K7).
-> jobs-cpu1.txt (colo), jobs-cpu2.txt (Thor)."""
import sys

def st(a, b, s): return ",".join(f"{a + i * s:g}" for i in range(int(round((b - a) / s)) + 1))

SIZES = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
LAMS = ["1e-08", "3e-08", "1e-07", "3e-07", "1e-06", "3e-06", "1e-05"]
PTS = "KN_SWJ_PTS=2:0.91,2.5:0.935,4:0.989,5:1.0"
NEW = "KN_V3R=1 KN_EMBT=2 KN_HEADSLAB=1 KN_LEVELS=../../data/v3/levels KN_DAMP=1 KN_TP=../../twopass-m2048.jsonl KN_PICKS=1"
ARMS = [   # name, env, sizes, accurate?, fast lambdas (as w58 ran them), box
    ("main", f"{NEW} KN_NOA4=1 KN_SWJ=all {PTS}", SIZES, 1, LAMS, "cpu2"),
    ("gu", f"{NEW} KN_NOA4=1 KN_SWJ=gu {PTS}", SIZES, 1, LAMS[3:], "cpu1"),
    ("noi8k7", f"{NEW} KN_NOA4=1 KN_SWJ=all {PTS} KN_I8={st(1.125, 5, 0.125)},{st(5.25, 6.75, 0.25)}", [6.5, 7.0], 0, LAMS[4:], "cpu1"),
]
out = {"cpu1": [], "cpu2": []}
for name, env, sizes, acc, lams, box in ARMS:
    for s in sorted(sizes, reverse=True):          # big sizes are the slow solves: start them first
        if acc: out[box].append(f"{name}-acc-{s:g}\t{env} KN_MODES=f16\t{s:g}\t0")
        for lm in lams: out[box].append(f"{name}-f-{s:g}-l{lm}\t{env}\t{s:g}\t{lm}")
for box, js in out.items():
    open(f"{sys.argv[1]}/jobs-{box}.txt", "w").write("\n".join(js) + "\n")
    print(box, len(js))
