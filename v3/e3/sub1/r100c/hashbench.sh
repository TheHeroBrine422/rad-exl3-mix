#!/bin/bash
# hashbench.sh -- #100c speed (r100c.md): w29's five exl3i dev builds of one tree (/data/tp/w29/b-VARIANT: old =
# mul1i8 today, cf = closed form (default), cfz, iqrh = IQR's hash in mul1i8's slot, abl = no decode ALU) on card 0,
# K2-5: bench_twopass M 2048 (a8 + phases: actq / wdec8 / gemm) and M 1, 4, 16 (f16 GEMV, nt_pass), 30 reps.
# Each pass holds /root/card0.lock (+ /root/ram.lock shared) ~1-2 min; it runs only if card 0 samples idle (amd-smi
# GFX_ACTIVITY <= 5% over 5 s: the user's processes run outside the lock) and counts only if the card is idle after it
# too (else its results move to res/p<i>-busy-*). Up to 3 passes in w29's three orders, no new pass after DEADLINE s
# (default 1020) from the start. Output /data/tp/w48/res/p<i>/<variant>-K<k>-{2048,gemv}.csv, log bench.log.
# Run as a q2.sh insert (/data/q1106/insert-30-hashbench.sh, hard cap 20 min). Never card 1.
W=/data/tp/w48; B29=/data/tp/w29; LOG=$W/bench.log; T0=$(date +%s); DL=${DEADLINE:-1020}
mkdir -p $W/res
log() { echo "$(date -u +%T) $*" >> $LOG; }
busy() { local m=0 a; for i in 1 2 3 4 5; do a=$(amd-smi metric -g 0 -u 2>/dev/null | awk '/GFX_ACTIVITY/{print $2}'); [ "${a:-100}" -gt $m ] && m=$a; sleep 1; done; echo $m; }
ORD=("old cf cfz iqrh abl|4 2 3 5" "abl iqrh cfz cf old|5 3 2 4" "cf iqrh old abl cfz|2 4 5 3")
for v in old cf cfz iqrh abl; do [ -x $B29/b-$v/bench_twopass ] && [ -f $B29/b-$v/radiance_home/kernels/exl3i.so ] || { log "missing build $v"; exit 1; }; done
log "== start (deadline ${DL} s)"
i=1
while [ $i -le 3 ] && [ $(( $(date +%s) - T0 )) -lt $DL ]; do
  (
    exec 5>/root/card0.lock; flock -w 300 5 || exit 3
    exec 7>/root/ram.lock; flock -s -w 120 7 || exit 3
    b=$(busy); log "pass $i: idle sample max GFX $b%"
    [ "$b" -gt 5 ] && exit 1
    export ROCR_VISIBLE_DEVICES=0
    o=$W/res/p$i; rm -rf $o; mkdir -p $o; ord=${ORD[$((i - 1))]}
    for K in ${ord#*|}; do for v in ${ord%|*}; do
      B=$B29/b-$v; f=$o/$v-K$K
      timeout 120 $B/bench_twopass $B/radiance_home/kernels/exl3i.so --rate $K --reps 30 --m 2048 --modes a8,phases --csv $f-2048.csv > $f-2048.log 2>&1
      timeout 120 $B/bench_twopass $B/radiance_home/kernels/exl3i.so --rate $K --reps 30 --m 1,4,16 --modes f16 --csv $f-gemv.csv > $f-gemv.log 2>&1
    done; done
    b=$(busy); log "pass $i done; idle sample after: max GFX $b%"
    [ "$b" -gt 5 ] && exit 2
    touch $o/.ok; exit 0
  ); rc=$?
  log "pass $i release rc=$rc"
  if [ $rc = 0 ]; then i=$((i + 1))
  else [ $rc = 2 ] && mv $W/res/p$i $W/res/p$i-busy-$(date +%s); sleep 45; fi
done
log "== end: $(ls -d $W/res/p? 2>/dev/null | wc -l) clean passes, $(( $(date +%s) - T0 )) s"
