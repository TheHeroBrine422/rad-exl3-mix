#!/bin/bash
# run_u.sh -- d1111 amendment 1 (ukisai arm + extra goldens); waits for run.sh, idempotent
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
X=$P/exp/d1111 X8=$P/exp/d1108
O=/data/yb/d1111 D8=/data/yb/d1108 D3=/data/yb/d1103; U=$O/ukisai
f32() { for d in $O $D8 $D3 $Y; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D8 $D3 $Y; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
while pgrep -f "^bash /data/yb/d1111/snap/[r]un.sh" > /dev/null; do sleep 120; done
C23=$Y/c23/hdiag-snap1024-dy GU=/data/g08-ukisai GT=/data/g08-agent-calib
if [ ! -f $GU/rows.f32 ]; then n=$(python3 -c "import json; print(len(json.load(open('$U/eval/kld.json'))['docs']))")
  lock; python3 $P/mkgold.py $M $U/eval $GU $(seq 0 $((n - 1))) > $O/gold-u.log 2>&1; unlock; log "g08-ukisai: $(tail -n 1 $O/gold-u.log | cut -c1-80)"; fi
[ -f $O/hinU/.done ] || { lock; python3 $P/hcollect.py $M $U/hin $O/hinU > $O/hinU.log 2>&1 && touch $O/hinU/.done; unlock; log "hin U: $(tail -n 1 $O/hinU.log)"; }
d=$O/hi-U0.25; [ -f $d/.done ] || python3 -I $X8/blend.py hin /data/h08/calib $O/hinU 0.25 $d >> $O/blend.log 2>&1
for s in 0 1; do for k in 3 2; do base=k${k}dygm$(sfx $s)
  lock; ev uk-$base $(wqd $base) $GU; ev tr-$base $(wqd $base) $GT; unlock
  t=k${k}hiU0.25$(sfx $s)
  lock; q $t $k i8 --hin $O/hi-U0.25 --hout $C23 --sigma-reg-out 0.01 --seed-offset $s; ev uk-$t $O/q-$t/wq $GU; ev tr-$t $O/q-$t/wq $GT; unlock
  pair $base $t; pair uk-$base uk-$t $GU; pair tr-$base tr-$t $GT
  for b in X0.25 X0.5 B0.25; do x=k${k}hi$b$(sfx $s); [ -d $O/q-$x/wq ] || continue
    lock; ev uk-$x $O/q-$x/wq $GU; ev tr-$x $O/q-$x/wq $GT; unlock; pair uk-$base uk-$x $GU; pair tr-$base tr-$x $GT; done
done; done
for b in U0.25 X0.25 X0.5 B0.25; do for k in 3 2; do python3 $X3/summ.py $O k${k}dygm k${k}hi$b uk-k${k}dygm uk-k${k}hi$b tr-k${k}dygm tr-k${k}hi$b | tee -a $O/result.txt; done; done
log "D1111U DONE"
