#!/bin/bash
# after.sh -- (ds8) #177-03 after stage C (lead note 11-13 7:30 PM CST: the ladder, then heads, then a/b donors): waits for
# "STAGE C DONE" in mad/logs/stageC.log, then the 57 mad R16 head donors (run151h.sh build mad/tools/q-mad.txt on HIP 0-6),
# their head-only KL on the fit + search dumps (run151h.sh score), the price table (rep151h.py -> mad/heads-rep/hcurve151.json),
# the wq drop, then the a/b T4 / T5 donors (ab151.sh). Every step is restartable and skips finished units; rerun after.sh
# to resume. Stop: touch /data/q27/mad/STOP (and heads/STOP for the head workers). Process name `bash /data/q27/mad/tools/after.sh`.
set -u; Q=/data/q27; T=$Q/mad/tools; H=$Q/heads; exec >> $Q/mad/logs/after.log 2>&1
say() { echo "$(date -u +%T) $*"; }
until grep -q "STAGE C DONE" $Q/mad/logs/stageC.log 2>/dev/null; do [ -f $Q/mad/STOP ] && { say STOP; exit 2; }; sleep 300; done
say "stage C done: heads"
bash $T/run151h.sh build $T/q-mad.txt 0 1 2 3 4 5 6
[ -f $Q/mad/STOP ] && { say STOP; exit 2; }
bash $T/run151h.sh score 6
W1=$H/w119; mkdir -p $Q/mad/heads-rep
env SC=$H/scores DOCS=$W1/code/v3/yaqa/w119/docs119.json SHARE=$W1/code/v3/yaqa/w119/calibshare119.json PERMS=$W1/perms \
  OUT=$Q/mad/heads-rep H140=$H/w146/rep/hcurve140.json R119=$W1/code/v3/yaqa/w119 $Q/venv/bin/python -I $T/rep151h.py \
  > $Q/mad/heads-rep/rep151h.out 2>&1 && say "table: $(grep 'price table' $Q/mad/heads-rep/rep151h.out | cut -c1-200)" || say "TABLE FAIL"
bash $T/run151h.sh drop
say "heads done: a/b"
bash $T/ab151.sh; say "AFTER DONE"
