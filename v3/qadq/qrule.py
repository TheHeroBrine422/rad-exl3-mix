#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qrule.py ROOT [--apply] -- the checkpoint rule of the QAD queue (user 11-22, README "Checkpoint rule"): every training line gives a CURVE: a j217 read at each of 0.25 M / 0.5 M / 1 M / 2 M / 4 M rows up to the
line's budget.  Plain lines (WSD): budgets = those points up to the final budget (each a decayed branch forked off the main run), reads = their checkpoints.  Lines with --lr-steps / --decay-shape (one budget; no constant
phase or piecewise lr): the run's end is the real number, the intermediate points are `--short-decay P1,P2,..` branches (fork at P, linear decay of the run's lr over 50 k rows: checkpoints ckpt-sd-<P>, labelled
short-decay in results.tsv). Not touched: lines already claimed / with a run dir, forks (--fork-from), --accum > 1 (the trainer allows one budget, no branches), M / B / R / U lines.
Without --apply: prints what would change; with it: rewrites jobs.txt under queue.lock (a backup jobs.txt.pre-qrule is kept)."""
import fcntl, os, shlex, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qq
root = sys.argv[1]; apply = "--apply" in sys.argv
PTS = [2.5e5, 5e5, 1e6, 2e6, 4e6]
ck = lambda x: "ckpt-%.0e" % x
FB = {2.5e5: "2.5e5", 5e5: "5e5", 1e6: "1e6", 2e6: "2e6", 4e6: "4e6", 1.5e6: "1.5e6", 3e6: "3e6"}
fb = lambda x: FB.get(x, "%g" % x)
lk = open(f"{root}/queue.lock", "a"); fcntl.flock(lk, fcntl.LOCK_EX)
lines = open(f"{root}/jobs.txt").read().split("\n"); out = []; ch = 0
for ln in lines:
    if not ln.startswith("T "): out.append(ln); continue
    j = qq.parse(ln, root)                                   # state checks only: the rewrite works on the raw tokens (placeholders {runs} / {ext} / {lam*x} stay)
    if j is None or j["kind"] != "T" or os.path.exists(f"{root}/claims/{j['key']}") or os.path.exists(f"{root}/runs/{j['arm']}") or os.path.exists(f"{root}/done/{j['key']}"): out.append(ln); continue
    raw = shlex.split(ln, comments=True); arm, size, bud, reads = raw[1:5]; rest = raw[5:]; dirs = [x for x in rest if x.startswith("@")]; fl = [x for x in rest if not x.startswith("@")]
    fin = float(bud.split(",")[-1])
    if "--fork-from" in fl or ("--accum" in fl and int(fl[fl.index("--accum") + 1]) > 1): out.append(ln); continue
    sched = "--lr-steps" in fl or ("--decay-shape" in fl)
    pts = [p for p in PTS if p < fin]; new_fl = list(fl)
    if sched:
        want_b = bud; sd = ",".join(fb(p) for p in pts)
        want_r = ",".join([f"ckpt-sd-{'%.0e' % p}" for p in pts] + [ck(fin)])
        if "--short-decay" in new_fl: i = new_fl.index("--short-decay"); new_fl[i + 1] = sd
        elif pts: new_fl += ["--short-decay", sd]
    else:
        want_b = ",".join(fb(p) for p in pts + [fin]); want_r = ",".join(ck(p) for p in pts + [fin])
    same = [float(x) for x in want_b.split(",")] == [float(x) for x in bud.split(",")] and want_r == reads and new_fl == fl
    if same: out.append(ln); continue
    nl = "T " + " ".join(shlex.quote(x) for x in [arm, size, want_b, want_r] + new_fl + dirs)
    print(f"{arm}: budgets {bud} -> {want_b}; reads {reads} -> {want_r}" + ("; + --short-decay" if sched and pts else "")); out.append(nl); ch += 1
if apply and ch:
    open(f"{root}/jobs.txt.pre-qrule", "w").write("\n".join(lines)); open(f"{root}/jobs.txt.new", "w").write("\n".join(out)); os.replace(f"{root}/jobs.txt.new", f"{root}/jobs.txt")
print(f"{ch} line(s) {'rewritten' if apply else 'would change'}")
