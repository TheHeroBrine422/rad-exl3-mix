#!/bin/bash
# s2c.sh PAR -- #177 stage 2c on colo: the fixed set's tiles (512, seed 12, K 1-6) at L 17 / 18 for mul1i8, the hash
# (A 0xd0474b), its Gaussian remap and cubic shaping; HYB (hash -> permuted Gaussian LUT) at L 16, IB 8 / 9 / 10; HYB LUTs
# fitted Lloyd-style (hfit.py, 256 tiles seed 21, 8 rounds) at K 2 / 4, IB 9 / 10, then scored on the fixed set. Resumable.
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p res logs luts; export OMP_NUM_THREADS=1
B=0x2545F491; A=0xd0474b
jobs() {
  for L in 17 18; do
    for s in m1i8 h:$A:$B:0x3F3F3F3F hg:$A:$B:0x3F3F3F3F hc:0.02:$A:$B:0x3F3F3F3F; do
      n=fx-$(echo $s | tr ':' '_')-L$L; o=res/$n.jsonl; [ -f $o ] && [ $(wc -l < $o) -ge 6 ] && continue
      echo "python3 hscreen.py eval $s 1,2,3,4,5,6 $L 512 12 $o > logs/$n.log 2>&1"
    done
  done
  for IB in 8 9 10; do
    s=hyb:$A:$B:$IB; n=fx-$(echo $s | tr ':' '_')-p; o=res/$n.jsonl; [ -f $o ] && [ $(wc -l < $o) -ge 6 ] && continue
    echo "python3 hscreen.py eval $s 1,2,3,4,5,6 16 512 12 $o > logs/$n.log 2>&1"
  done
  for IB in 9 10; do for K in 2 4; do
    f=luts/hyb-$IB-K$K.npy; n=fx-hybfit-$IB-K$K; o=res/$n.jsonl; [ -f $o ] && [ $(wc -l < $o) -ge 1 ] && continue
    echo "( [ -f $f ] || python3 hfit.py $A $B $IB $K 256 21 8 $f > logs/fit-$IB-K$K.log 2>&1 ) && python3 hscreen.py eval hyb:$A:$B:$IB:$f $K 16 512 12 $o > logs/$n.log 2>&1"
  done; done
}
jobs | xargs -P ${1:-16} -I{} bash -c "nice -n 19 {}"
echo S2C DONE
