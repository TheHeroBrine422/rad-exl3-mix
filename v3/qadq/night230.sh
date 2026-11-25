#!/bin/bash
# night230.sh (box2; lead 11-25 ~5:40 AM CST, user asleep): (1) cont files for the proportion screens now; (2) when w230's chain releases the
# cards: w231's DP gate (4 cards) + branch check (1 card); then the screens on cards 0,1 via the queue while w231's p2p stress + deep300
# run on cards 2,3. Log /data/q27/queue/q.log (prefix night230).
set -u
Q=/data/q27/queue; L=$Q/q.log; PY=/data/q27/venv/bin/python; C=$Q/cont228; W31=/data/q27/w231
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') night230: $*" >> $L; }
say "start"
if [ ! -e $C/cont-p402030.json ]; then
$PY - <<PYEOF >> $L 2>&1 || { say "cont build failed: stop"; exit 1; }
import json, sys
import numpy as np
sys.path.insert(0, "$Q/code/v3/qad181"); import data181 as D
T = 1552745
cal = json.load(open("$C/cont-calib.json"))["items"]; xg = json.load(open("$C/cont-xg1.json"))["items"]
g = D.Gold("$Q/ext/gold/tset2", genfile="$Q/ext/genmask/tset2.u8"); n = len(json.load(open("$Q/ext/gold/tset2/kld.json"))["docs"])
te = []
for i in np.random.default_rng(0).permutation(n):
    r = g.nrows(int(i), 8192)
    if r >= 16: te.append(["tset2", int(i), int(r)])
print("night230: tset2 rows at cap 8192", sum(x[2] for x in te), "docs", len(te))
def take(it, t):
    out, s = [], 0
    for x in it:
        if s >= t: break
        out.append(x); s += x[2]
    return out, s
for name, (c, x, t) in {"p502525": (50, 25, 25), "p602515": (60, 25, 15), "p403030": (40, 30, 30), "p701515": (70, 15, 15)}.items():
    a, sa = take(te, T * t // 100); b, sb = take(xg, T * x // 100); d, sd = take(cal, T - sa - sb)
    json.dump({"goldens": ["xg-scr1", "tset2"], "seed": 0, "items": a + b + d}, open("$C/cont-" + name + ".json", "w"))
    print("night230:", name, "teacher", sa, "xgap", sb, "calib", sd)
open("$C/cont-p402030.json", "w").write("{}")
PYEOF
say "cont files written"
fi
until grep -q "run230: chain exited" /data/q27/teach230/chain.log; do sleep 120; done
sleep 60; say "w230 chain exited ($(tail -1 /data/q27/teach230/chain.log | cut -c1-80)); cards free"
for c in 0 1 2 3; do echo NONE > $Q/only-card$c; done
cd $W31
code/v3/qad231/gate231.sh code/v3/qad231/gpu231b.txt 0 1 2 3 > gate-b.out 2>&1; say "w231 DP gate (gpu231b) rc $?"
code/v3/qad231/gate231.sh code/v3/qad231/gpu231c.txt 0 > gate-c.out 2>&1; say "w231 branch check (gpu231c) rc $?"
rm -f $Q/only-card0 $Q/only-card1
A="--lr 3e-4 --lr-steps 3e-4:1.5e6,1e-4:1.11e7 --plan-ext {ext}/ext-plan.json --ext-root {ext} --compile-scales --compile-ew --a8-int8"
F="--fork-from $Q/runs-ice/I25-step-10M/state.pt"
( exec 9>$Q/queue.lock; flock 9; cp $Q/jobs.txt $Q/jobs.txt.pre-night230
  { echo "# lead 11-25 (night230): proportion screens calib : xgap : teacher(set 2, cap 8192) from the 10M state, 1.55M rows each + teacher-set-2 judge (JX) baselines"
    for b in "I25-step-10M $Q/runs-ice/I25-step-10M/ckpt-1e+07" "C228-ctl $Q/runs-ice/C228-ctl/ckpt-1e+07" "C229-xgc C229-xgc/ckpt-1e+07" "C228-opmix C228-opmix/ckpt-1e+07"; do set -- $b; echo "R JX-$1 2.50 $2"; done
    for p in p502525 p602515 p403030 p701515; do
      echo "T C230-$p 2.50 1.15e7 ckpt-1e+07 $F --cont $C/cont-$p.json $A"
      for j in JC JG JX JL; do echo "R $j-C230-$p 2.50 C230-$p/ckpt-1e+07"; done
    done; } >> $Q/jobs.txt )
say "screens queued (cards 0,1); w231 p2p stress + deep300 on cards 2,3"
code/v3/qad231/gate231.sh code/v3/qad231/gpu231a.txt 2 3 > gate-a.out 2>&1; say "w231 p2p stress + deep300 (gpu231a) rc $?"
rm -f $Q/only-card2 $Q/only-card3; say "cards 2,3 back to the queue; done"
