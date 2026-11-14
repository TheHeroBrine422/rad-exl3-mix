#!/bin/bash
# trimloop.sh [ROOT] -- (ds8) every 5 min, keeps the mad ladder's backlog small (11-13 ~11:45 PM CST: ds8 free space fell
# ~150 GB / h while the swj passes trail the quant lanes):
#   (1) a level with a phase-1 DONE and an assembled level/: its out/*/t-*.safetensors go (level/ holds the same codes; the
#       swj redo and post read level/, nothing re-assembles) -> PRETRIM-OUT
#   (2) a level with K <= 4.5 whose DONE has no scales yet (its swj pass is to come) and no swj claim: its phase-1 final/ and
#       final.recipe go (C9-only; the swj redo deletes and rebuilds them anyway) -> PRETRIM-FINAL. A claimed level is never
#       touched (its swj pass may be writing the new final/).
#   (3) trim.sh for the levels that are current.
# Stops when ROOT/../logs shows "STAGE C DONE" or on /data/q27/mad/STOP. Log mad/logs/trimloop.log. (#177-03, w151)
R=${1:-/data/q27/ladder-mad}; Q=/data/q27; PY=$Q/venv/bin/python; LG=$Q/mad/logs/trimloop.log
say() { echo "$(date -u +%T) $*" >> $LG; }
while [ ! -f $Q/mad/STOP ]; do
  for D in $R/q27v3-*-mad; do
    [ -f $D/DONE ] && [ -f $D/level/stamp.json ] || continue
    n=$(basename $D); K=${n##*-k}; K=${K%-mad}
    if [ ! -f $D/PRETRIM-OUT ] && [ ! -f $D/TRIMMED ]; then
      b=$(du -sm $D | cut -f1); rm -f $D/out/*/t-*.safetensors; touch $D/PRETRIM-OUT; say "$n: out t-files removed ($b -> $(du -sm $D | cut -f1) MB)"
    fi
    if [ -d $D/final ] && [ ! -d $R/swjclaim/$n ] && \
       $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if float(sys.argv[2]) <= 4.5 and not d.get('scales') else 1)" $D/DONE $K 2>/dev/null; then
      [ -d $R/swjclaim/$n ] && continue           # re-check right before the delete
      b=$(du -sm $D | cut -f1); rm -rf $D/final $D/final.recipe; touch $D/PRETRIM-FINAL; say "$n: phase-1 final/ removed ($b -> $(du -sm $D | cut -f1) MB)"
    fi
  done
  bash $Q/mad/tools/trim.sh $R >> $LG 2>&1
  grep -q "STAGE C DONE" $Q/mad/logs/stageC.log 2>/dev/null && { say "stage C done: exit"; exit 0; }
  sleep 300
done
