#!/bin/bash
# chain226.sh (box2; setsid nohup, log /data/q27/w226/chain.log) -- w226 source 2: the bf16 golden of our deep session traces, then the D256-25T plan.
# Restartable: every step is skipped when its output exists.
#   1. wait until T2-D256-25 and T2-D256-30 are done (training + reads; lead 11-23: record after both)
#   2. stop the four dispatchers between jobs (stop-card<N>), wait until they exited
#   3. TP4 radiance --kld-record of QROOT/ext/trdeep/trdeep-b1.jsonl (12 pieces, 236k-250k tokens; bf16 KV, top-512 as tr32k-b1) into
#      QROOT/ext/gold/trdeep-b1 (.part, renamed with .done); all four card locks held (order 0..3, then ram shared)
#   4. the dispatchers restarted (qwait.sh), whatever step 3 gave
#   5. gen mask (genmask181.py make) -> QROOT/ext/genmask/trdeep-b1.u8
#   6. plan ext-plan-226t.json (plan226-build.py: the 26 calib deep docs + traces 0..2 of trdeep-b1; the D256-25T line waits for it)
# PRIVATE: trdeep-b1 is the user's traces (box2 counts as ours); never copy it or its golden elsewhere.
set -u
Q=/data/q27; QROOT=$Q/queue; W=$Q/w226; E=$QROOT/ext; L=$W/chain.log
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
source $QROOT/code/v3/qadq/qenv.sh box2
G=$E/gold/trdeep-b1
log "chain226 start (pid $$)"
until [ -e $QROOT/done/T2-D256-25 ] && [ -e $QROOT/done/T2-D256-30 ]; do sleep 120; done
if [ ! -e $G/.done ]; then
  log "D256-25 / D256-30 done: stopping the dispatchers between jobs"
  P=""; for c in 0 1 2 3; do P="$P $(cat $QROOT/dispatch/card$c.pid)"; touch $QROOT/stop-card$c; done
  for p in $P; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
  log "dispatchers stopped ($P): recording trdeep-b1 on TP4"
  AV=$(df -BG --output=avail $E | tail -1 | tr -dc 0-9)
  if [ "$AV" -lt 150 ]; then log "only $AV GB free: not recording"; else
    ( exec 5>$(CARDLOCK 0) 6>$(CARDLOCK 1) 7>$(CARDLOCK 2) 8>$(CARDLOCK 3) 9>$RAMLOCK; flock 5; flock 6; flock 7; flock 8; flock -s 9
      rm -rf $G.part; mkdir -p $(dirname $G); Y=$W/rec-trdeep-b1.yaml
      cat > $Y <<YEOF
memory:
  host-headroom-mib: 8192
models:
  trdeep-b1:
    model: $MODEL
    tp: 4
    kv-cache-dtype: bf16
    max-model-len: 262144
    max-num-seqs: 1
    kld-record: $G.part
    kld-corpus: $E/trdeep/trdeep-b1.jsonl
    kld-top: 512
YEOF
      s=$(date +%s); LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=0,1,2,3 $RB --config $Y > $W/rec-trdeep-b1.log 2>&1; rc=$?
      log "record rc $rc in $(( ($(date +%s) - s) / 60 )) min"
      if [ $rc = 0 ] && [ -e $G.part/kld.json ]; then cp $E/trdeep/trdeep-b1.jsonl.meta.jsonl $G.part/corpus.meta.jsonl; touch $G.part/.done; mv $G.part $G; fi )
  fi
  for c in 0 1 2 3; do setsid nohup $QROOT/code/v3/qadq/qwait.sh box2 $c < /dev/null > $QROOT/logs/qwait-$c-chain226.out 2>&1 & done
  log "dispatchers restarted"
fi
[ -e $G/.done ] || { log "no golden: stop"; exit 1; }
if [ ! -e $E/genmask/trdeep-b1.u8 ]; then
  $PY $QROOT/code/v3/qad181/genmask181.py make $G $E/genmask/trdeep-b1.u8.part $MODEL >> $L 2>&1 && mv $E/genmask/trdeep-b1.u8.part $E/genmask/trdeep-b1.u8
  log "gen mask rc $?"
fi
if [ ! -e $E/ext-plan-226t.json ]; then
  $PY $QROOT/code/v3/qad226/plan226-build.py $E/ext-plan.json $CALIB $W/deep2-ids.json $E/ext-plan-226t.json.part --gold trdeep-b1=$G:$E/genmask/trdeep-b1.u8 >> $L 2>&1 \
    && mv $E/ext-plan-226t.json.part $E/ext-plan-226t.json; log "plan ext-plan-226t rc $?"
fi
log "chain226 done"
