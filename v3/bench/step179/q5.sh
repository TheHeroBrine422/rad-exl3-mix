#!/bin/bash
# q5.sh -- card 1, after q4: is the step's GEMV slowdown (+2% every exl3i launch, step2 vs base) the power cap? ABBA
# order, K2.5, 1200-token decodes at ctx 0 stamped one pass in 16, with cardclock.py watching the SMU: base / step2 / step2
# / base on the mul1i8 pack (w138), mbase / mstep / mstep / mbase on the mad pack (w150). -> res/st/ab-*. (w152)
W=/data/q27/w152; cd $W; Q=/data/q27
while pgrep -f "^bash tools/q4[.]sh" > /dev/null; do sleep 20; done
say() { echo "$(date -u +%T) q5 $*" >> logs/q.log; }
ab() {   # ab PACKPATH A B
  local m=$1 pn=$(basename $1 .rad) i=0
  for h in $2 $3 $3 $2; do i=$((i + 1))
    NTOK=1200 STAMP_N=16 WATCHS=48 bash tools/stampw.sh 1 $W/b10$h/home $m res/st/ab-$i-$h-$pn
  done; say "ab $pn $2 $3"
}
ab $Q/w138/packs/v3-k2.5-r16k2.5-a8.rad base step2
ab $Q/w150/packs/q27v3-635bac02-k2.5-mad-a8.rad mbase mstep
say DONE
