#!/bin/bash
# s2d.sh PAR -- #177 stage 2c's HYB fits (s2c.sh's last four jobs, whose subshell form xargs could not run): hfit.py at K 2 / 4,
# IB 9 / 10 (256 tiles seed 21, 8 rounds), then the fitted LUT on the fixed set. Resumable. (w139)
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p res logs luts; export OMP_NUM_THREADS=1
B=0x2545F491; A=0xd0474b
for IB in 9 10; do for K in 2 4; do
  f=luts/hyb-$IB-K$K.npy; n=fx-hybfit-$IB-K$K; o=res/$n.jsonl; [ -f $o ] && [ $(wc -l < $o) -ge 1 ] && continue
  echo "bash -c '[ -f $f ] || python3 hfit.py $A $B $IB $K 256 21 8 $f > logs/fit-$IB-K$K.log 2>&1; python3 hscreen.py eval hyb:$A:$B:$IB:$f $K 16 512 12 $o > logs/$n.log 2>&1'"
done; done | xargs -P ${1:-4} -I{} bash -c "nice -n 19 {}"
echo S2D DONE
