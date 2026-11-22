#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""hours.py JOBS.txt [...] -- estimated card-hours of the T lines of a queue file (w222): w220's compiled speeds (2.50, other cards busy) + 10 % for A8: full model 98 s / step, layers 32-63 67 s, 48-63 50 s
(x1.1 with --prompt-weight), ~21.2 k gen rows a step (the plan holds 2.52 M rows, ~21 k a step up to ~2 M), + 7 min per served read. A run's steps = last budget + 0.2 x every earlier budget (the WSD branches).
Prints per line and the total; the int8 kernel (w223) should make the steps cheaper (reads unchanged): the totals are the conservative side."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qq

ROWS_STEP, READ_H = 21200.0, 7 / 60


def hours(j):
    b = [float(x) for x in j["budgets"].split(",")]
    rows = b[-1] + sum(0.2 * x for x in b[:-1])
    fl = j["flags"]; t = 98.0
    if "--short-decay" in fl: rows += 5e4 * len(fl[fl.index("--short-decay") + 1].split(","))     # w222: each short-decay branch = ~50 k rows
    if "--fork-from" in fl:                                   # a fork resumes at its state's rows (state-<rows>.pt)
        import re
        m = re.search(r"state-(\d+)\.pt", fl[fl.index("--fork-from") + 1]); rows -= int(m.group(1)) if m else 0
    if "--local-mse" in fl:                                   # w222 guesses: block stage 1 (the KL is cut: layers after the last tap run forward only) 0.65 / 0.8 of a step for taps to 31 / 47; joint +5 %
        spec = fl[fl.index("--local-mse") + 1].split(",")
        if spec[-1] == "joint": t *= 1.05
        elif "--train-layers" in fl: pass          # (0-31 / 0-47 factors are in the train-layers table below)
    if "--train-layers" in fl:
        t = {"32-63": 67.0, "48-63": 50.0, "0-31": 98.0 * 0.65, "0-47": 98.0 * 0.8}.get(fl[fl.index("--train-layers") + 1], 98.0)
    if "--prompt-weight" in fl: t *= 1.1
    return rows / ROWS_STEP * t / 3600, len(j["reads"]) * READ_H


def main():
    tot = 0.0
    for f in sys.argv[1:]:
        os.makedirs("/tmp/hours_q", exist_ok=True); open("/tmp/hours_q/jobs.txt", "w").write(open(f).read())
        s = 0.0
        for j in qq.jobs("/tmp/hours_q"):
            if j["kind"] != "T" or "--max-steps" in j["flags"]: continue
            tr, rd = hours(j); s += tr + rd; print(f"{j['arm']:16s} {j['size']} train {tr:5.2f} h + reads {rd:4.2f} h = {tr + rd:5.2f}")
        print(f"== {f}: {s:.1f} card-h"); tot += s
    if len(sys.argv) > 2: print(f"== total {tot:.1f} card-h")


if __name__ == "__main__": main()
