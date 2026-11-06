#!/bin/bash
# repair_v.sh -- after run_v.sh: recompute every paired ratio of amendment 3 into result.txt (where summ.py reads; run_v
# logged to result-v.txt and its pair() clobbered arm()'s $b, so most xc / ag / vf pair lines are empty), then summarize.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
O=/data/yb/d1111 D8=/data/yb/d1108 D3=/data/yb/d1103
f32() { for d in $O $D8 $D3 $Y; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
while pgrep -f "^bash /data/yb/d1111/snap/[r]un_v.sh" > /dev/null; do sleep 120; done
GX=/data/g08-xcorp2 GA=$D8/g-agent GV=/data/g08-verify1
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
for k in 2 3 4 5; do for s in 0 1 2; do b=k${k}dygm$(sfx $s); [ -f $(f32 $b) ] || continue
  for h in X0.25 X0.5 X0.75 X1.0 X3_0.5 code-0.5 prose-0.5 mt-0.5; do t=k${k}hi$h$(sfx $s); [ -f $(f32 $t) ] || continue
    grep -q "pair $b -> $t: \"ratio\"" $O/result.txt || pair $b $t
    for p in xc:$GX ag:$GA vf:$GV; do x=${p%%:*} g=${p#*:}; [ -f $(f32 $x-$t) ] && [ -f $(f32 $x-$b) ] || continue
      grep -q "pair $x-$b -> $x-$t: \"ratio\"" $O/result.txt || pair $x-$b $x-$t $g; done; done; done; done
for h in X0.25 X0.5 X0.75 X1.0 X3_0.5 code-0.5 prose-0.5 mt-0.5; do for k in 3 2 4 5; do
  python3 $X3/summ.py $O k${k}dygm k${k}hi$h vf-k${k}dygm vf-k${k}hi$h xc-k${k}dygm xc-k${k}hi$h ag-k${k}dygm ag-k${k}hi$h; done; done \
  | grep -v "no pairs" | tee $O/summary-v.txt
log "D1111 REPAIR DONE"
