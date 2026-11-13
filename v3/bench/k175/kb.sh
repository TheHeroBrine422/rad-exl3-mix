#!/bin/bash
# kb.sh CARD ROUNDS -- #175 (w137) kernel bench of the asm-literal fix: bench_twopass (w124's kpre / kdec forms) of b10b
# (today's device code) vs b10v (the fix), interleaved per round, K4.125 and K2.5: decode (f16, M 1 / 4) and the A8
# prefill path (phases, M 512 / 2048). Under card CARD's lock. CSV + txt into res/kb/. (w137)
C=$1; N=${2:-3}; W=/data/q27/w137; O=$W/res/kb; mkdir -p $O
export HIP_VISIBLE_DEVICES=$C LD_LIBRARY_PATH=/opt/rocm-10.1/lib
exec 9>/data/q27/locks/card$C.lock; flock 9
for r in $(seq 1 $N); do
  for k in 4.125 2.5; do
    for b in b10b b10v; do
      B=$W/$b/pb/exl3i; S=$W/$b/home/kernels/exl3i.so
      [ -f $O/r$r-$b-k$k-dec.csv ] || $B/bench_twopass $S --rate $k --m 1,4 --modes f16 --reps 50 --csv $O/r$r-$b-k$k-dec.csv > $O/r$r-$b-k$k-dec.txt 2>&1 || echo "FAIL dec r$r $b k$k"
      [ -f $O/r$r-$b-k$k-pre.csv ] || $B/bench_twopass $S --rate $k --m 512,2048 --modes phases --reps 20 --csv $O/r$r-$b-k$k-pre.csv > $O/r$r-$b-k$k-pre.txt 2>&1 || echo "FAIL pre r$r $b k$k"
    done
  done
done
echo "kb done $(date -u +%T)"
