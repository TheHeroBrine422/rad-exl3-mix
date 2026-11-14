#!/bin/bash
# after2.sh -- (ds8) #177-03: after.sh's head steps started as soon as the 45 mad head donors exist (11-14 7:40 AM CST; the
# quant cards were free while the last swj passes ran on HIP 5-6): head-only KL of every unscored R16s003* donor on the fit +
# search dumps (run151h.sh score on HIP 0), the price table (rep151h.py -> mad/heads-rep/hcurve151.json), the wq drop. The a/b
# donors (ab151.sh) run beside it on HIP 1-4. Restartable. Log mad/logs/after.log. Process name `bash /data/q27/mad/tools/after2.sh`.
set -u; Q=/data/q27; T=$Q/mad/tools; H=$Q/heads; exec >> $Q/mad/logs/after.log 2>&1
say() { echo "$(date -u +%T) $*"; }
n=$(ls $H/q/R16s003mad/K*/wq/lm_head.rowerr.npz 2>/dev/null | wc -l); [ $n -ge 45 ] || { say "only $n of 45 head donors"; exit 1; }
bash $T/ab151.sh "1 2 3 4" &
say "heads: score"; bash $T/run151h.sh score 0
W1=$H/w119; mkdir -p $Q/mad/heads-rep
env SC=$H/scores DOCS=$W1/code/v3/yaqa/w119/docs119.json SHARE=$W1/code/v3/yaqa/w119/calibshare119.json PERMS=$W1/perms \
  OUT=$Q/mad/heads-rep H140=$H/w146/rep/hcurve140.json R119=$W1/code/v3/yaqa/w119 $Q/venv/bin/python -I $T/rep151h.py \
  > $Q/mad/heads-rep/rep151h.out 2>&1 && say "table: $(grep 'price table' $Q/mad/heads-rep/rep151h.out | cut -c1-200)" || say "TABLE FAIL"
bash $T/run151h.sh drop
wait; say "AFTER2 DONE"
