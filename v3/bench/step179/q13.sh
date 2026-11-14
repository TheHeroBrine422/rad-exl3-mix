#!/bin/bash
# q13.sh -- card 1 (lead note 11-13 11:00 PM CST): re-measure #177-02's mad decode numbers that ran on the KS counter race,
# on qc-exl3-v3 fdd480c (home b10mfix = 9f0a059 + the same patch): e2e (w150's e2b settings) i8 twin / mad nt_pass (f16
# rows) / mad GEMV with its pre-pass / mad GEMV on xb rows / + #179 (b10fres2), K2.625 and K4.125, 2 rounds interleaved;
# decode-path KLD (A8, GEMV through EXL3I_MAD_SPLIT, f16 rows) per K, paired by w150's cmp115.py. (w152)
W=/data/q27/w152; cd $W; Q=/data/q27; H=$W/b10mfix/home; HS=$W/b10fres2/home
say() { echo "$(date -u +%T) q13 $*" >> logs/q.log; }
for K in 2.625 4.125; do k=${K/./}
  MP=$Q/w150/packs/q27v3-144dd6e1-k$K-mad-a8.rad; IP=$Q/w139/s3/q27v3-144dd6e1-k$K-i8/mix-a8.rad
  bash tools/kldenv.sh 1 k$k-a8=$H=$MP k$k-gemv=$H=$MP=EXL3I_MAD_SPLIT=1,EXL3I_A8_MIN_ROWS=1000000 \
    k$k-f16=$H=$MP=EXL3I_A8_MIN_ROWS=1000000,EXL3I_MAD_GEMV=0 >> logs/kldr.log 2>&1
done; say "kld"
for r in 1 2; do for K in 2.625 4.125; do k=${K/./}
  MP=$Q/w150/packs/q27v3-144dd6e1-k$K-mad-a8.rad; IP=$Q/w139/s3/q27v3-144dd6e1-k$K-i8/mix-a8.rad
  bash tools/e2b.sh 1 $r i8-$k=$H=$IP mad$k-f16=$H=$MP=QWEN35_EXL3_AQ_XB=0,EXL3I_MAD_GEMV=0 mad$k-pre=$H=$MP=QWEN35_EXL3_AQ_XB=0 \
    mad$k-xb=$H=$MP mad$k-xb-step=$HS=$MP
done; done
say DONE
