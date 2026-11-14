#!/bin/bash
# stageB.sh -- (ds8) #177-03 stage B (lead note 11-13 7:30 PM CST): the full mad levels K2.5, K4, K6 in the ladder root
# /data/q27/ladder-mad: quant on HIP 0-6 (HIP 7 is w153's) with steps jobs / assemble / post / recipe, then one swj
# pass over K2.5 + K4 (swj_batch 2; K6 is above post.swj.k_max) on the first free card of 0-6, steps assemble / swj /
# post / recipe (rxrun redoes swj + post on the DONE-without-scales levels). Restartable (runloop.sh). Process names:
# `bash /data/q27/mad/tools/stageB.sh`, `bash .../runloop.sh`, `python -m rxrun.run /data/q27/mad/conf/*.json`.
set -u; Q=/data/q27; T=$Q/mad/tools; K=$Q/mad/conf; mkdir -p $K $Q/mad/logs; exec >> $Q/mad/logs/stageB.log 2>&1
say() { echo "$(date -u +%T) $*"; }
PY=$Q/venv/bin/python; R=$Q/ladder-mad
$PY $T/mkconf.py $K/qB.json --root $R --cards 0,1,2,3,4,5,6 --levels 2.5-mad,4-mad,6-mad
say "quant start"; bash $T/runloop.sh $K/qB.json qB || { say "quant FAILED"; exit 1; }
for c in 0 1 2 3 4 5 6; do flock -n $Q/locks/card$c.lock true && { CARD=$c; break; }; done
$PY $T/mkconf.py $K/sB.json --root $R --cards ${CARD:-0} --kinds swj,quant --steps assemble,swj,post,recipe --levels 2.5-mad,4-mad
say "swj start (card ${CARD:-0})"; bash $T/runloop.sh $K/sB.json sB 10 || { say "swj FAILED"; exit 1; }
say "STAGE B LEVELS DONE"
