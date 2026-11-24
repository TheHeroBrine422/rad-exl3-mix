#!/bin/bash
# gold228.sh NAME (box2; log /data/q27/onpol/w228/gold.log) -- w228 step 5: the bf16 golden of the on-policy documents.
# NAME = onpol-b1 (the trace batch: render-b + out) | onpol-calib-b1 (calib chat: render-calib-chat + out-calib-chat, then calib text:
# render-calib-text + out-calib-text). Restartable: every step is skipped when its output exists. Run after the generators ended.
#   1. stop the two generation servers (serve0.pid / serve1.pid)
#   2. docs228.py -> QROOT/ext/onpol/NAME.jsonl (+ .meta.jsonl, .stats.json)
#   3. TP4 radiance --kld-record (bf16 model, bf16 KV, top-512 as xc8k-b1 / trdeep-b1, the queue's radiance build as chain226.sh) into
#      QROOT/ext/gold/NAME.part; then gen.u8 (all ones: the corpus' score_from makes every row a generated token), corpus.meta.jsonl,
#      .done, renamed to NAME; the gen mask QROOT/ext/genmask/NAME.u8 (same bytes, the --plan-ext layout)
#   4. check228.py (rows, tokens) and a 6-document bf16-vs-itself --kld-ref (golden_pick.py slice) -> W/selfcheck-NAME.json
# The four card locks are held by hold.sh (w228) for the whole run. PRIVATE: the user's sessions; never copied off box2.
set -u
Q=/data/q27; QROOT=$Q/queue; W=$Q/onpol/w228; E=$QROOT/ext; L=$W/gold.log; C=$W/code/v3/qad228
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
source $QROOT/code/v3/qadq/qenv.sh box2
NAME=${1:?NAME}; G=$E/gold/$NAME; CORP=$E/onpol/$NAME.jsonl
case $NAME in
  onpol-b1) PARTS="--part $W/render-b $W/out pi-onpol";;
  onpol-calib-b1) PARTS="--part $W/render-calib-chat $W/out-calib-chat calib-onpol --part $W/render-calib-text $W/out-calib-text calib-onpol";;
  *) echo "unknown $NAME"; exit 2;;
esac
log "gold228 $NAME start (pid $$)"
for f in $W/gen*.pid; do kill -0 $(cat $f) 2>/dev/null && { log "a generator ($f) still running: stop"; exit 1; }; done
for i in 0 1; do p=$(cat $W/serve$i.pid 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && { kill $p; log "server $i (pid $p) stopped"; }; done
for i in 0 1; do p=$(cat $W/serve$i.pid 2>/dev/null); [ -n "$p" ] && while kill -0 $p 2>/dev/null; do sleep 5; done; done
mkdir -p $E/onpol $E/genmask
if [ ! -e $CORP ]; then
  $PY -I $C/docs228.py $E/onpol/$NAME $PARTS >> $L 2>&1; log "docs228 $NAME rc $?"
fi
[ -e $CORP ] || { log "no corpus: stop"; exit 1; }
if [ ! -e $G/.done ]; then
  AV=$(df -BG --output=avail $E | tail -1 | tr -dc 0-9)
  [ "$AV" -lt 150 ] && { log "only $AV GB free: not recording"; exit 1; }
  rm -rf $G.part; Y=$W/rec-$NAME.yaml
  cat > $Y <<YEOF
memory:
  host-headroom-mib: 8192
models:
  $NAME:
    model: $MODEL
    tp: 4
    kv-cache-dtype: bf16
    max-model-len: 9216
    max-num-seqs: 4
    kld-record: $G.part
    kld-corpus: $CORP
    kld-top: 512
YEOF
  s=$(date +%s); LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=0,1,2,3 $RB --config $Y > $W/rec-$NAME.log 2>&1; rc=$?
  log "record $NAME rc $rc in $(( ($(date +%s) - s) / 60 )) min"
  if [ $rc = 0 ] && [ -e $G.part/kld.json ]; then
    $PY -c "import json,numpy as np; m=json.load(open('$G.part/kld.json')); np.ones(m['rows'],np.uint8).tofile('$G.part/gen.u8')"
    cp $E/onpol/$NAME.jsonl.meta.jsonl $G.part/corpus.meta.jsonl; touch $G.part/.done; mv $G.part $G
  fi
fi
[ -e $G/.done ] || { log "no golden: stop"; exit 1; }
[ -e $E/genmask/$NAME.u8 ] || { cp $G/gen.u8 $E/genmask/$NAME.u8.part && mv $E/genmask/$NAME.u8.part $E/genmask/$NAME.u8; log "gen mask $NAME written"; }
$PY -I $C/check228.py $G $CORP $MODEL >> $L 2>&1; log "check228 $NAME rc $?"
SC=$W/selfcheck-$NAME.json
if [ ! -e $SC ]; then
  S=$W/selfslice-$NAME; rm -rf $S; N=$($PY -c "import json; print(len(json.load(open('$G/kld.json'))['docs']))")
  $PY -I $W/code/eval/golden_pick.py $G $CORP $S/gold $S/corpus.jsonl 0 1 2 $((N / 2)) $((N - 2)) $((N - 1)) >> $L 2>&1
  cat > $W/self-$NAME.yaml <<YEOF
memory:
  host-headroom-mib: 8192
models:
  self:
    model: $MODEL
    tp: 4
    kv-cache-dtype: bf16
    max-model-len: 9216
    max-num-seqs: 4
    kld-ref: $S/gold
    kld-out: $SC.part
YEOF
  LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=0,1,2,3 $RB --config $W/self-$NAME.yaml > $W/selfcheck-$NAME.log 2>&1; log "self-check $NAME rc $?"
  [ -e $SC.part ] && mv $SC.part $SC
fi
log "gold228 $NAME done"
