#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""tl233.py RUN_DIR [STEP_INDEX] -- w233: read the per-segment timelines of a --pipe-procs run (W233_TL=1: OUT/tl233-r<rank>.jsonl, one line a
step a stage) and report, per stage and step: GPU spans of the forwards / backwards (event to event on the card), the idle time between them (at the
step start, inside, at the end), and the host lag (how long after the previous GPU span ended the stage's host reached the next span's start:
>0 = the card waited for its own host). Segment fields: sid, item, sa, sb, hist, f0 gpu, f0 host, f1 gpu, f1 host, b0 gpu, b0 host, b1 gpu, b1 host
(GPU times relative to the step's first event on the card, host times relative to its record)."""
import glob, json, os, sys

run = sys.argv[1]
files = sorted(glob.glob(os.path.join(run, "tl233-r*.jsonl")))
steps = {}
for f in files:
    r = int(f.rsplit("-r", 1)[1].split(".")[0])
    for k, l in enumerate(open(f)):
        steps.setdefault(k, {})[r] = json.loads(l)
pick = [int(sys.argv[2])] if len(sys.argv) > 2 else sorted(steps)
for k in pick:
    st = steps[k]; T0 = min(z["t_host"] for z in st.values())
    print(f"== step index {k}")
    for r, z in sorted(st.items()):
        off = z["t_host"] - T0
        iv = []
        for s in z["segs"]:
            iv.append((s[5], s[7], "F", s, s[6]))
            if s[9] is not None: iv.append((s[9], s[11], "B", s, s[10]))
        iv.sort(key=lambda x: x[0])
        busy = sum(b - a for a, b, *_ in iv); end = max(b for _, b, *_ in iv); start = min(a for a, *_ in iv)
        gaps = [(iv[j + 1][0] - iv[j][1], iv[j], iv[j + 1]) for j in range(len(iv) - 1)]
        idle_in = sum(max(0.0, g) for g, *_ in gaps)
        hostlag = sum(max(0.0, nx[4] - prv[1]) for g, prv, nx in gaps if g > 0.005)   # the host reached the next span after the card finished the previous
        nF = sum(1 for x in iv if x[2] == "F"); nB = len(iv) - nF
        fF = sum(b - a for a, b, t, *_ in iv if t == "F"); fB = busy - fF
        print(f"  stage {z['stage']} (rank {r}, starts {off:+.2f} s): spans {busy:.2f} s (F {fF:.2f} in {nF}, B {fB:.2f} in {nB}) of {end:.2f} s; "
              f"idle before first {start:.2f}, inside {idle_in:.2f} (gaps > 5 ms: {sum(1 for g, *_ in gaps if g > 0.005)}; host lag in them {hostlag:.2f})")
        big = sorted(gaps, key=lambda x: -x[0])[:5]
        for g, prv, nx in big:
            print(f"     gap {g:.3f} s after {prv[2]} of seg {prv[3][:4]} (ends {prv[1]:.2f}) before {nx[2]} of seg {nx[3][:4]} (host reached it at {nx[4]:.2f})")
