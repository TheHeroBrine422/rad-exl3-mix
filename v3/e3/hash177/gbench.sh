#!/bin/bash
# gbench.sh TAG ROUNDS "BUILDS" [MLIST] [RATES] [MODES] -- per-shape exl3i timing (bench_twopass, random codes at each
# rate) of hbuild.sh's plugin variants, builds interleaved per (round, rate), on one box2 card under its lock (the first
# free of CARDS, default "3 2 1") + ram.lock shared. rocm-smi state (clocks, power, temperature, use) of that card before
# and after every round. CSV per (round, build, rate) into /data/q27/w139/bench/TAG/. (w139, #177)
T=$1; N=$2; B=$3; MS=${4:-1,8}; RATES=${5:-"1.5 2 2.5 3 4 5 6"}; MODES=${6:-f16}
H=/data/q27/w139; L=/data/q27/locks; O=$H/bench/$T; mkdir -p $O
exec >> $O/bench.log 2>&1
say() { echo "$(date -u +%T) $*"; }
while :; do
  for c in ${CARDS:-3 2 1}; do
    exec 5>$L/card$c.lock
    if flock -n 5; then CARD=$c; break 2; fi
    exec 5>&-
  done
  sleep 20
done
exec 4>$L/ram.lock; flock -s 4
export ROCR_VISIBLE_DEVICES=$CARD HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-10.1/lib
G=$((3 - CARD))   # rocm-smi index of HIP device CARD (box2.md)
smi() { rocm-smi -d $G --showuse --showpower --showtemp --showclocks --showmemuse 2>/dev/null | grep -E "GPU\[" | grep -viE "^$" > $O/smi-$1.txt; }
say "card $CARD (rocm-smi $G) locked, rounds $N, builds $B, M $MS, rates $RATES, modes $MODES"
for r in $(seq 1 $N); do
  smi r$r-before
  for k in $RATES; do
    for b in $B; do
      $H/pb/$b/bench_twopass $H/pb/$b/radiance_home/kernels/exl3i.so --rate $k --m $MS --modes $MODES --reps ${REPS:-50} \
        --csv $O/r$r-$b-k$k.csv > $O/r$r-$b-k$k.txt 2>&1 || say "FAIL r$r $b k$k"
    done
  done
  smi r$r-after
  say "round $r done"
done
say "unlock card $CARD"
