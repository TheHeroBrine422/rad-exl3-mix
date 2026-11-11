#!/bin/bash
# dev.sh TAG TOOL NDOCS SRCSPEC [extra swjfit args] -- one swj level pass on the truncated dev model (#144, w109)
#   TOOL: ref (python -m rxq.swjfit, today's code) | a .py path (a candidate copy); SRCSPEC: names, e.g. "fs0 fs1" or
#   "fs0 fs1 fs0:b fs1:b" (NAME:SUFFIX = the same level dir under another name)
# Holds the box's card0.lock for the run. Out: $R/TAG/{out,scr,run.log,wall.json}; a finished TAG is skipped.
set -u
W=${W:-/data/w109}; R=$W/runs; M=${MODEL:-$W/m8}; META=$W/meta/swj27-qad; LV=${LV:-$W/lv8}
TAG=$1 TOOL=$2 ND=$3 SPEC=$4; shift 4
. $W/env.sh; [ -n "${PP:-}" ] && export PYTHONPATH=$PP   # PP: another pipeline tree (a patched rxq)
D=$R/$TAG; [ -f $D/wall.json ] && { echo "$TAG done"; exit 0; }
rm -rf $D; mkdir -p $D
SRC=(); for x in $SPEC; do n=${x/:/_}; SRC+=(--src "$n=$LV/${x%%:*}:level"); done
if [ "$TOOL" = ref ]; then CMD=(python3 -m rxq.swjfit); else CMD=(python3 $TOOL); fi
t0=$(date +%s.%N)
( exec 9>/root/card0.lock; flock 9; t1=$(date +%s.%N); echo "lock wait $(awk "BEGIN{print $t1 - $t0}")" > $D/lock.txt
  "${CMD[@]}" $M $META $D/out "${SRC[@]}" --scratch $D/scr --maxdocs $ND --fused --no-cache --grid 0.6:1.4:0.01 "$@" > $D/run.log 2>&1
  rc=$?; t2=$(date +%s.%N)
  echo "{\"tag\": \"$TAG\", \"rc\": $rc, \"wall\": $(awk "BEGIN{print $t2 - $t1}"), \"tool\": \"$TOOL\", \"docs\": $ND, \"src\": \"$SPEC\", \"extra\": \"$*\"}" > $D/wall.json.part
  [ $rc = 0 ] && mv $D/wall.json.part $D/wall.json; exit $rc )
