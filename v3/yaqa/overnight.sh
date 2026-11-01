#!/bin/bash
# overnight.sh -- 2026-11-01 night, unattended (log /data/yaqa/overnight.log):
#  1. gate 2e EARLY: wait for the collector's 100-sequence snapshot (hin-snap100), stop gate2e.sh and the collector,
#     quantize hin100-K2 (card 0) / hin100-K3 (card 1), evaluate both on golden-ver1, paired vs one-sided (y2-1s-K*).
#     Pass rule (fixed before the run): PASS = diff < 0 and z <= -2 at K2 AND K3; FAIL = diff > 0 at either K, or no
#     win at either; anything else MARGINAL (-> a 200/400-sequence collection later, with the user).
#  2. d2.sh (split-half input Hessians, both cards), 3. a4time.sh (card 0, ~10 min),
#  4. the V3 level ladder on both cards (levels.sh, disjoint lists; resumes K3.25 at layer 16).
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa F=/data/yaqa/full64 QD=/data/yaqa/qe PY=/data/venv-exl3/bin/python
BASE=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad
log() { echo "$(date -u +%H:%M) $*" | tee -a $Y/overnight.log; }
cd $RXM_WORK
# ---- 1. early gate 2e
if ! grep -q "gate2e done" $Y/gate2e.log 2>/dev/null; then
  until grep -q "^seq 100:" $Y/full64.log && [ $(ls $F/hin-snap100/*.safetensors 2>/dev/null | wc -l) -ge 8 ] \
        && ! ls $F/hin-snap100/*.part > /dev/null 2>&1; do
    grep -q "^DONE" $Y/full64.log && break
    pgrep -f "^/data/venv-exl3/bin/python -I collect.py /data/yaqa/full64" > /dev/null || { log "collector died before snap100: stop"; exit 1; }
    sleep 30; done
  log "snapshot 100: $(ls $F/hin-snap100 | wc -l) H_in files; stopping gate2e.sh, d2.sh, a4wait.sh, the collector"
  for p in "^/bin/bash ./gate2e.sh" "^/bin/bash ./d2.sh" "^/bin/bash ./a4wait.sh"; do pkill -f "$p"; done
  pkill -f "^/data/venv-exl3/bin/python -I collect.py /data/yaqa/full64"
  while pgrep -f "^/data/venv-exl3/bin/python -I collect.py /data/yaqa/full64" > /dev/null; do sleep 5; done
  sleep 10
  export EXL3_QT_OPTIMIZED=1; mkdir -p $QD
  q() {  # CARD VARIANT K
    [ -f $QD/$2/q-L23.safetensors ] && return
    ( exec 9>$RXM_WORK/card$1.lock; flock 9
      ROCR_VISIBLE_DEVICES=$1 $PY -I ourq/ourq.py -K $3 -o $QD/$2 --layers 22-23 --device 0 --hin $F/hin-snap100 2>&1 \
        | grep -v -e Warning -e "cpu = _conv" ) > $QD/$2.log 2>&1
    grep -q '^done' $QD/$2.log && log "$2 quantized" || log "$2 quantize FAIL"
  }
  q 0 hin100-K2 2 & q 1 hin100-K3 3 & wait
  T22="ffn_down ffn_gate_up ssm_out ssm_qkv ssm_z"; T23="attn_k attn_output attn_qg attn_v ffn_down ffn_gate_up"
  G=""; for t in $T22; do G="$G blk.22.$t.weight"; done; for t in $T23; do G="$G blk.23.$t.weight"; done
  Q=$Y/gate2e.q; : > $Q; V=""
  for v in hin100-K2 hin100-K3; do
    [ -f $QD/$v/q-L23.safetensors ] || continue
    ./ourq/ourconv.sh $QD/$v.rad sc4.00-e4 $QD/$v $G >> $Y/overnight.log 2>&1 || { log "$v conv FAIL"; continue; }
    : > $QD/$v.mix; for g in $G; do echo "$g $QD/$v.rad" >> $QD/$v.mix; done
    echo "y6-$v $BASE $QD/$v.mix golden-ver1 corpus/ver1.jsonl" >> $Q; V="$V $v"
  done
  echo END >> $Q
  ./auto/evalq2.sh 0 $Q >> $Y/gate2e-eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $Y/gate2e-eval1.log 2>&1 & wait
  S=$RXM_WORK/evq/scores; R=""
  for v in $V; do
    K=${v##*-K}; [ -f $S/y6-$v.f32 ] || { log "$v: no score"; continue; }
    r=$($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-ver1 $S/y2-1s-K$K.f32 $S/y6-$v.f32)
    log "$v vs one-sided: $r"; echo "$v vs one-sided: $r" >> $Y/gate2e.log; R="$R
$r"
  done
  VERD=$(echo "$R" | $PY -I -c "
import json, sys
d = [json.loads(l[l.index('{'):]) for l in sys.stdin if '{' in l]
if len(d) < 2: print('INCOMPLETE')
elif all(x['diff'] < 0 and x['z'] <= -2 for x in d): print('PASS')
elif any(x['diff'] > 0 for x in d) or not any(x['diff'] < 0 for x in d): print('FAIL')
else: print('MARGINAL')")
  log "GATE 2E (100 seqs) VERDICT: $VERD"
  rm -f $QD/*.rad
  echo "$(date -u +%H:%M) gate2e done (early, 100 sequences): $VERD" >> $Y/gate2e.log
fi
# ---- 2. D2, 3. A4 timing
( cd $Y && ./d2.sh ) >> $Y/overnight.log 2>&1
log "d2: $(tail -1 /data/d2/d2.log)"
( exec 9>$RXM_WORK/card0.lock; flock 9
  ROCR_VISIBLE_DEVICES=0 /data/bench/a4time.sh /data/bench/a4time.jsonl >> /data/bench/a4time.log 2>&1
  python3 /data/bench/a4report.py /data/bench/a4time.jsonl /data/bench/speedcal-stock.jsonl >> /data/bench/a4time.log 2>&1 )
log "a4: $(grep -c '"ms"' /data/bench/a4time.jsonl 2>/dev/null) points"
# ---- 4. levels on both cards
/data/v3levels/levels.sh 0 "3.25 3.75 2.5 4.25 2.25 4.75 5 6" > /dev/null 2>&1 &
/data/v3levels/levels.sh 1 "2.75 3.5 3 4 4.5 2 5.5" > /dev/null 2>&1 &
log "levels started on both cards"
wait
log "overnight done"
