#!/bin/bash
# run147b.sh -- (w147, #178-02, box2) the isolation scores, after run147.sh has released its card (one card at a time): the
# first free card, held to the end, ram.lock shared; golden-s171d, box2 tools/score.sh, W/home.
#   T4aq0  W/mix/2.5-T4.rad with QWEN35_EXL3_AQ=0: every consumer quantises h itself with its own suh (no AQ site). Rows
#          equal to T4's = the fused weight reads its own plane of the site (the wiring), not qkv's / z's.
#   T4o    a copy of 2.5-B.rad with bf16 a / b = the oracle's decode of the served records (abcheck147.py OUT, w142's
#          abpatch.py): the served weight as a dequantized splice; T4 - T4o = the A8 activation of a / b (+ kernel order).
#   T4s    the same with w142's own ab-T4.safetensors: w142's arm rebuilt here (= ds8's T4 rows if the hosts agree).
# Copies deleted after their score. Restartable.
set -u
Q=/data/q27 W=/data/q27/w147 LK=/data/q27/locks G=$W/gold/golden-s171d
say() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/run147.log; }
while ! grep -q "run147 finished" $W/run147.log; do sleep 30; done
while :; do
  C=
  for cc in 0 1 2 3; do exec 9>$LK/card$cc.lock; if flock -n 9; then C=$cc; break; fi; exec 9>&-; done
  [ -n "$C" ] && break; sleep 5
done
exec 6>$LK/ram.lock; flock -s 6
say "run147b: card $C held (pid $$)"
o=$W/scores/2.5-T4aq0-s171d
[ -f $o.json.rows ] || { QWEN35_EXL3_AQ=0 bash $Q/tools/score.sh $Q/rad10-perf/install $W/home $W/mix/2.5-T4.rad $G $o $C | tee -a $W/run147.log; }
for v in T4o:$W/ab-T4oracle.safetensors T4s:$W/ab-T4.safetensors; do
  n=${v%%:*} ab=${v#*:} o=$W/scores/2.5-$n-s171d r=$W/mix/2.5-$n.rad
  [ -f $o.json.rows ] && continue
  rm -f $r; cp --reflink=auto $W/mix/2.5-B.rad $r && /data/q27/venv/bin/python -I $W/abpatch.py $r $ab >> $W/run147.log 2>&1 \
    && bash $Q/tools/score.sh $Q/rad10-perf/install $W/home $r $G $o $C | tee -a $W/run147.log || say "$n FAILED"
  rm -f $r
done
exec 9>&-
say "run147b finished"
