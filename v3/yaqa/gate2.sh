#!/bin/bash
# gate2.sh -- YAQA gate 2: does the output-side Hessian lower the real error of K2/K3 tensors?
# After collect.py's pilot (ROWS, 200 qad sequences): fold H_out for blocks 22 (linear attention) and 23 (full attention),
# quantize their linears at K2 and K3 one-sided (1s, today's quantizer) and two-sided (2s, same seeds), swap each set
# into the V2 6.00 release (11 tensors) and score base + 4 variants on the verify slice golden-ver1 with evalq2.sh.
# Gate: 2s's KLD increment over base 15-30% below 1s's, paired (mix/paired.py).
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK
Y=/data/yaqa ROWS=${ROWS:-/data/yaqa/rows} HO=$Y/hout-pilot QD=$Y/q
BASE=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad
PY=/data/venv-exl3/bin/python
log() { echo "$(date -u +%H:%M) $*" | tee -a $Y/gate2.log; }
cd ${RXM_WORK}
until grep -q "^DONE" $Y/pilot.log; do sleep 30; done
log "pilot done: $(ls $ROWS/.done | wc -l) sequences"
FOLD_THREADS=14 $PY -I $Y/fold.py $ROWS $HO 22,23 >> $Y/gate2.log 2>&1 || { log "fold FAIL"; exit 1; }
log "folded $(ls $HO | wc -l) H_out"
export EXL3_QT_OPTIMIZED=1
quant() {  # CARD VARIANT [ourq args]
  local c=$1 v=$2; shift 2
  for K in 2 3; do
    ( exec 9>${RXM_WORK}/card$c.lock; flock 9
      $PY -I ourq/ourq.py -K $K -o $QD/$v-K$K --layers 22-23 --device 0 "$@" 2>&1 | grep -v -e Warning -e "cpu = _conv" ) \
      > $QD/$v-K$K.log 2>&1 && grep -q '^done' $QD/$v-K$K.log || { log "$v-K$K quantize FAIL"; continue; }
    log "$v-K$K quantized"
  done
}
mkdir -p $QD
ROCR_VISIBLE_DEVICES=0 quant 0 1s & ROCR_VISIBLE_DEVICES=1 quant 1 2s --hout $HO & wait
T22="ffn_down ffn_gate_up ssm_out ssm_qkv ssm_z"; T23="attn_k attn_output attn_qg attn_v ffn_down ffn_gate_up"
G=""; for t in $T22; do G="$G blk.22.$t.weight"; done; for t in $T23; do G="$G blk.23.$t.weight"; done
Q=$Y/gate2.q; : > $Q
echo "y2-base $BASE - golden-ver1 corpus/ver1.jsonl" >> $Q
for v in 1s-K2 2s-K2 1s-K3 2s-K3; do
  [ -f $QD/$v/q-L23.safetensors ] || continue
  ./ourq/ourconv.sh $QD/$v.rad sc4.00-e4 $QD/$v $G >> $Y/gate2.log 2>&1 || { log "$v conv FAIL"; continue; }
  : > $QD/$v.mix; for g in $G; do echo "$g $QD/$v.rad" >> $QD/$v.mix; done
  echo "y2-$v $BASE $QD/$v.mix golden-ver1 corpus/ver1.jsonl" >> $Q
  log "$v converted, queued"
done
echo END >> $Q
./auto/evalq2.sh 0 $Q >> $Y/gate2-eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $Y/gate2-eval1.log 2>&1 & wait
grep '"tag": "y2-' evq/results.jsonl | tee -a $Y/gate2.log
log "gate2 done"
