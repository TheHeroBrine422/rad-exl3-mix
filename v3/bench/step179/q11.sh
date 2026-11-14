#!/bin/bash
# q11.sh -- card 1, after q9: the last arriver's batched write (fres2 / mres2 = 67aab95 worktree; fres2 + kctr-key.patch):
# exl3i device tests of mres2 (10.1, 7.14); KLD rows fres2 (def, c64; mad K2.5, mul1i8 K2.5); gdec fres2 (both mad
# packs); stamps d0 / d8 (mad K2.5) fres2; e2e rounds 8, 9: fbase / fstep / fres2 on both mad packs, step2 / fres2 on
# mul1i8 K2.5 (8k prefill). (w152)
W=/data/q27/w152; cd $W; Q=/data/q27
while pgrep -f "^bash tools/q9[.]sh" > /dev/null; do sleep 20; done
say() { echo "$(date -u +%T) q11 $*" >> logs/q.log; }
bash tools/ctall.sh 1 10 mres2; bash tools/ctall.sh 1 714 mres2; say "ctall $(grep -c PASS res/gt/10mres2-ctall.txt) / $(grep -c PASS res/gt/714mres2-ctall.txt) PASS"
MK=$Q/w150/packs/q27v3-635bac02-k2.5-mad-a8.rad; M4=$Q/w150/packs/q27v3-144dd6e1-k4.125-mad-a8.rad
bash tools/kld.sh 1 $MK fres2-def fres2-c64 >> logs/kld.log 2>&1
bash tools/kld.sh 1 v3-k2.5-r16k2.5-a8 fres2-def fres2-c64 >> logs/kld.log 2>&1; say "kld"
for m in $MK $M4; do bash tools/gdec.sh 1 $m fres2 $W/b10fres2/home; done; say "gdec"
bash tools/stamp.sh 1 $W/b10fres2/home $MK res/st/fres2-$(basename $MK .rad)-d0
bash tools/stamp.sh 1 $W/b10fres2/home $MK res/st/fres2-$(basename $MK .rad)-d8 $Q/w137/prompts/ctx-8k.txt
for h in mfix fstep; do bash tools/stamp.sh 1 $W/b10$h/home $MK res/st/$h-$(basename $MK .rad)-d8 $Q/w137/prompts/ctx-8k.txt; done
say "stamp"
for r in 8 9; do
  ROWS="fbase=$W/b10mfix/home fstep=$W/b10fstep/home fres2=$W/b10fres2/home" bash tools/e2.sh 1 $r $MK $M4
  ROWS="step2=$W/b10step2/home fres2=$W/b10fres2/home" bash tools/e2.sh 1 $r v3-k2.5-r16k2.5-a8
done
say DONE
