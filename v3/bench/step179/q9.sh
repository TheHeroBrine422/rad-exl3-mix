#!/bin/bash
# q9.sh -- card 1, after qnext (re-planned 11-13 10:35 PM CST: mad decode needs kctr-key.patch, so the mad rows use the
# f* homes = their tree + the patch: fbase = mfix (qc-exl3-v3 9f0a059), fstep (b73675c), fres (b73675c + exl3i version 5,
# actq's residual add)). exl3i device tests of mres (10.1, 7.14); KLD rows; gdec; stamps d0; e2e rounds 6, 7. (w152)
W=/data/q27/w152; cd $W; Q=/data/q27
while pgrep -f "^bash tools/q(next|[4-8])[.]sh" > /dev/null; do sleep 20; done
say() { echo "$(date -u +%T) q9 $*" >> logs/q.log; }
bash tools/ctall.sh 1 10 mres; bash tools/ctall.sh 1 714 mres; say "ctall"
MK=$Q/w150/packs/q27v3-635bac02-k2.5-mad-a8.rad; M4=$Q/w150/packs/q27v3-144dd6e1-k4.125-mad-a8.rad
bash tools/kld.sh 1 $MK fres-def fres-c64 >> logs/kld.log 2>&1
bash tools/kld.sh 1 v3-k2.5-r16k2.5-a8 fres-def fres-c64 >> logs/kld.log 2>&1
bash tools/kld.sh 1 $MK mres-714-def >> logs/kld.log 2>&1; say "kld"
for m in $MK $M4; do for h in fstep fres; do bash tools/gdec.sh 1 $m $h $W/b10$h/home; done; done; say "gdec"
for h in mfix fstep fres; do bash tools/stamp.sh 1 $W/b10$h/home $MK res/st/$h-$(basename $MK .rad)-d0; done
bash tools/stamp.sh 1 $W/b10fres/home $Q/w138/packs/v3-k2.5-r16k2.5-a8.rad res/st/fres-v3-k2.5-r16k2.5-a8-d0; say "stamp"
for r in 6 7; do
  ROWS="fbase=$W/b10mfix/home fstep=$W/b10fstep/home fres=$W/b10fres/home" bash tools/e2.sh 1 $r $MK $M4
  ROWS="step=$W/b10step2/home fres=$W/b10fres/home" bash tools/e2.sh 1 $r v3-k2.5-r16k2.5-a8
done
say DONE
