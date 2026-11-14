#!/bin/bash
# stageC.sh [QCARDS] [SWJCARDS] -- (ds8) #177-03 stage C (lead note 11-13 7:30 PM CST): the other 46 mad levels in
# /data/q27/ladder-mad, focus sizes first: G1 K2.625-4.5, G2 K0.125-2.375 (low K first), G3 K4.625-7.5. Quant (steps
# jobs / assemble / post / recipe) on QCARDS (default 0-6; never 7 = w153's), one rxrun.run a group; the swj passes run
# beside it (swjC.sh, one loop per SWJCARDS card, default "5 6"), claiming finished levels in pairs. Each quant attempt
# re-makes its conf without the levels a swj loop has claimed (a restarted quant run must not redo a level's post while
# its swj redo runs). Writes ladder-mad/QUANT-DONE at the end (the swj loops then drain and exit), then trim.sh.
# Restartable; stop: touch /data/q27/mad/STOP. Process names `bash /data/q27/mad/tools/stageC.sh`, `.../swjC.sh`,
# `python -m rxrun.run /data/q27/mad/conf/*.json`. (w151)
set -u; Q=/data/q27; T=$Q/mad/tools; R=$Q/ladder-mad; K=$Q/mad/conf; PY=$Q/venv/bin/python
QC=${1:-0,1,2,3,4,5,6}; SC=${2:-5 6}; mkdir -p $K $Q/mad/logs; exec >> $Q/mad/logs/stageC.log 2>&1
say() { echo "$(date -u +%T) $*"; }
case ",$QC," in *,7,*) say "HIP 7 is w153's"; exit 1;; esac
G1="2.625 2.75 2.875 3 3.125 3.25 3.375 3.5 3.625 3.75 3.875 4.125 4.25 4.375 4.5"
G2="0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 2 2.125 2.25 2.375"
G3="4.625 4.75 4.875 5 5.25 5.5 5.75 6.25 6.5 6.75 7 7.5"
rm -f $R/QUANT-DONE
for c in $SC; do pgrep -f "^bash $T/swjC.sh $c\$" > /dev/null || { (setsid nohup bash $T/swjC.sh $c > /dev/null 2>&1 < /dev/null &); say "swj loop on card $c"; }; done
for g in G1 G2 G3; do
  for a in 1 2 3 4 5 6; do
    [ -f $Q/mad/STOP ] && { say "STOP"; exit 2; }
    lv=""; for k in ${!g}; do n=q27v3-635bac02-k$k-mad; [ -d $R/swjclaim/$n ] || [ -f $R/$n/DONE ] || lv="$lv,$k-mad"; done
    [ -z "$lv" ] && { say "$g: every level DONE or with swj"; break; }
    $PY $T/mkconf.py $K/q$g.json --root $R --cards $QC --levels ${lv#,} > /dev/null
    say "$g attempt $a: ${lv#,}"
    env PYTHONPATH=$Q/stage/code/pipeline $PY -m rxrun.run $K/q$g.json >> $Q/mad/logs/q$g.log 2>&1 && { say "$g OK"; break; }
    say "$g attempt $a rc $?"; sleep 30
  done
  bash $T/trim.sh $R
done
touch $R/QUANT-DONE; say "QUANT-DONE"
while pgrep -f "^bash $T/swjC.sh" > /dev/null; do sleep 120; bash $T/trim.sh $R; done
bash $T/trim.sh $R; say "STAGE C DONE"
