#!/bin/bash
# stageD.sh -- (ds8) #177-03: the sub-1 mad rebuild resumed after ds8's crash (11-14 ~10:00 AM CST), on the cards other work leaves
# free (LANES, default "3,4 5"; the menu work holds the rest). stageC3's lane / claim machinery without the HIP 5-6 lane: quant
# first (only the sub-1 levels are left: every K >= 1 level is DONE), QUANT-DONE, then the swj passes (swjC.sh loops, pairs) on
# SWJCARDS (default "3 4 5") and trim.sh, "STAGE D DONE". Stale claims of unfinished levels (qclaim, and swjclaim of levels whose
# DONE has no scales: their pass died with the machine) are cleared at the start. Restartable; stop: touch /data/q27/mad/STOP.
# Process names `bash /data/q27/mad/tools/stageD.sh [lane CARDS]`, `.../swjC.sh N`. (w151)
set -u; Q=/data/q27; T=$Q/mad/tools; R=$Q/ladder-mad; PY=$Q/venv/bin/python; LG=$Q/mad/logs/stageC.log
LANES=${LANES:-"3,4 5"}; SC=${SWJCARDS:-"3 4 5"}
say() { echo "$(date -u +%T) $*" >> $LG; }
if [ "${1:-}" = lane ]; then exec bash $T/stageC3.sh lane $2; fi          # stageC3's lane function (claims, rxrun.run, status)
mkdir -p $R/qclaim $R/swjclaim; rm -f $R/QUANT-DONE
for d in $R/qclaim/*; do [ -d "$d" ] && [ ! -f $R/$(basename $d)/DONE ] && rmdir $d; done
for d in $R/swjclaim/*; do n=$(basename $d); [ -d "$d" ] || continue
  grep -q '"scales": "' $R/$n/DONE 2>/dev/null || { rmdir $d; rm -rf $R/swj-*/$n; say "stageD: stale swj claim + partial swj of $n cleared"; }; done
say "stageD: quant lanes $LANES, then swj on $SC"
for c in $LANES; do bash $T/stageD.sh lane $c & done; wait
[ -f $Q/mad/STOP ] && { say "STOP"; exit 2; }
touch $R/QUANT-DONE; say "stageD: QUANT-DONE"
for c in $SC; do (setsid nohup bash $T/swjC.sh $c > /dev/null 2>&1 < /dev/null &); done
sleep 30; while pgrep -f "^bash $T/swjC.sh" > /dev/null; do sleep 120; done
bash $T/trim.sh $R >> $LG 2>&1; say "STAGE D DONE"
