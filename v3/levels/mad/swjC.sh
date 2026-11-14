#!/bin/bash
# swjC.sh CARD -- (ds8) the mad ladder's swj passes on one card, beside the quant run (stageC.sh): repeatedly claims up to
# two levels (mkdir ROOT/swjclaim/<level>) that are DONE without scales (phase 1: jobs / assemble / post / recipe) and
# need swj (K <= 4.5), lowest K first, waits up to 20 min for a second one while quant runs, then runs rxrun.run with
# steps assemble / swj / post / recipe on CARD (swj_batch 2; rxrun redoes swj + post on those levels; the card lock is
# shared with the quant pool through flock), retried by runloop.sh, then trim.sh. Exits when stageC.sh has finished
# (ROOT/QUANT-DONE) and nothing is left. Stop: touch /data/q27/mad/STOP. Process name `bash /data/q27/mad/tools/swjC.sh`.
set -u; Q=/data/q27; T=$Q/mad/tools; R=$Q/ladder-mad; C=$1; PY=$Q/venv/bin/python; LG=$Q/mad/logs/swjC-c$C.log
mkdir -p $R/swjclaim $Q/mad/conf; exec >> $LG 2>&1
say() { echo "$(date -u +%T) $*"; }
[ $C = 7 ] && { say "HIP 7 is w153's"; exit 1; }
ready() {   # levels needing swj, phase-1 DONE, unclaimed, lowest K first
  for D in $R/q27v3-*-mad; do
    [ -f $D/DONE ] && [ ! -d $R/swjclaim/$(basename $D) ] || continue
    K=${D##*-k}; K=${K%-mad}
    $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if float(sys.argv[2]) <= 4.5 and not d.get('scales') else 1)" $D/DONE $K \
      && echo "$K $(basename $D)"
  done | sort -g | cut -d' ' -f2
}
say "start on card $C"
while [ ! -f $Q/mad/STOP ]; do
  lv=$(ready | head -n 2)
  if [ $(echo "$lv" | grep -c .) -lt 2 ] && [ ! -f $R/QUANT-DONE ]; then
    w=0; while [ $(ready | grep -c .) -lt 2 ] && [ ! -f $R/QUANT-DONE ] && [ $w -lt 40 ]; do sleep 30; w=$((w + 1)); done
    lv=$(ready | head -n 2)
  fi
  if [ -z "$lv" ]; then [ -f $R/QUANT-DONE ] && break; sleep 60; continue; fi
  got=""; for n in $lv; do mkdir $R/swjclaim/$n 2>/dev/null && got="$got $n"; done
  [ -z "$got" ] && continue
  tag=$(for n in $got; do k=${n##*-k}; echo -n "${k%-mad}+"; done); tag=${tag%+}
  $PY $T/mkconf.py $Q/mad/conf/swj-$tag.json --root $R --cards $C --kinds swj,quant --steps assemble,swj,post,recipe \
    --levels $(echo $got | tr ' ' ',') > /dev/null
  say "swj $tag start"
  if bash $T/runloop.sh $Q/mad/conf/swj-$tag.json swj-$tag 10; then say "swj $tag OK"; bash $T/trim.sh $R
  else say "swj $tag FAILED (claims kept: a rerun of this level needs rm -r $R/swjclaim/<level>)"; fi
done
say "exit (QUANT-DONE $( [ -f $R/QUANT-DONE ] && echo yes || echo no ), STOP $( [ -f $Q/mad/STOP ] && echo yes || echo no ))"
