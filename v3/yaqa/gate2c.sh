#!/bin/bash
# gate2c.sh -- YAQA gate 2, full-token H_out: the last test before dropping YAQA (gate 2b: fp32 sampled-position H_out
# was still worse than one-sided). Same 200 qad windows and the same sampled labels as the fp32 pilot (rows32), but
# H_out = Σ over EVERY position (collect.py YAQA_FULL=22,23, fp32, backward truncated at layer 22, CPU sums).
#   0. truncation check: 2 sequences with the backward run to the embedding vs the main run's 2-sequence snapshot
#   1. collection (both cards)  2. two-sided K2/K3 at sigma_reg_out 0.025 and 0.3  3. swap into V2 6.00, golden-ver1,
#   paired vs gate2's one-sided quants (y2-1s-K*) and vs gate2b's sampled two-sided (y3-*).
# Waits for the head-slab test and the fractional-K smoke test (fracrun.sh), which hold the cards / RAM tonight.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa F=/data/yaqa/full32 HO=/data/yaqa/full32/hout QD=/data/yaqa/qfull
BASE=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad
PY=/data/venv-exl3/bin/python
log() { echo "$(date -u +%H:%M) $*" | tee -a $Y/gate2c.log; }
while pgrep -f "^/bin/bash (/data/yaqa/headrun.sh|/data/exl3frac/e3/fracrun.sh)" > /dev/null; do sleep 60; done
cd ${RXM_WORK}
export YAQA_FP32=1 YAQA_START=0 YAQA_FULL=22,23 YAQA_LABELS=$Y/rows32 TRITON_CACHE_AUTOTUNING=1
COL="exec 8>${RXM_WORK}/card0.lock 9>${RXM_WORK}/card1.lock 7>${RXM_WORK}/ram.lock; flock 8; flock 9; flock -s 7"
log "start"
if [ ! -f $HO/model.language_model.layers.23.self_attn.v_proj.safetensors ]; then
  ( eval "$COL"; cd $Y && YAQA_SNAP=2 $PY -I collect.py $F 200 2048 32 ) > $Y/full32.log 2>&1
  grep -q "^DONE" $Y/full32.log || { log "full-token collection FAIL"; exit 1; }
fi
log "collected: $(grep -c '^seq' $Y/full32.log) sequences, $(grep '^DONE' $Y/full32.log)"
if [ ! -f $Y/full32chk.done ]; then       # same 2 sequences, backward to the embedding
  ( eval "$COL"; cd $Y && YAQA_TRUNC=0 YAQA_SAVE_EVERY=1000 $PY -I collect.py $Y/full32chk 2 2048 32 ) > $Y/full32chk.log 2>&1
  $PY -I $Y/hcmp.py $F/hout-snap $Y/full32chk/hout >> $Y/gate2c.log 2>&1 && touch $Y/full32chk.done
  log "truncation check: $(tail -1 $Y/gate2c.log)"
fi
export EXL3_QT_OPTIMIZED=1
quant() {
  local c=$1; shift
  while [ $# -gt 0 ]; do
    local v=$1 K=$2 so=$3; shift 3
    [ -f $QD/$v/q-L23.safetensors ] && continue
    ( exec 9>${RXM_WORK}/card$c.lock; flock 9
      $PY -I ourq/ourq.py -K $K -o $QD/$v --layers 22-23 --device 0 --hout $HO --sigma-reg-out $so 2>&1 \
        | grep -v -e Warning -e "cpu = _conv" ) > $QD/$v.log 2>&1 && grep -q '^done' $QD/$v.log || { log "$v quantize FAIL"; continue; }
    log "$v quantized"
  done
}
mkdir -p $QD
ROCR_VISIBLE_DEVICES=0 quant 0 full-K2 2 0.025 fullr-K3 3 0.3 &
ROCR_VISIBLE_DEVICES=1 quant 1 full-K3 3 0.025 fullr-K2 2 0.3 &
wait
T22="ffn_down ffn_gate_up ssm_out ssm_qkv ssm_z"; T23="attn_k attn_output attn_qg attn_v ffn_down ffn_gate_up"
G=""; for t in $T22; do G="$G blk.22.$t.weight"; done; for t in $T23; do G="$G blk.23.$t.weight"; done
Q=$Y/gate2c.q; : > $Q
for v in full-K2 fullr-K2 full-K3 fullr-K3; do
  [ -f $QD/$v/q-L23.safetensors ] || continue
  ./ourq/ourconv.sh $QD/$v.rad sc4.00-e4 $QD/$v $G >> $Y/gate2c.log 2>&1 || { log "$v conv FAIL"; continue; }
  : > $QD/$v.mix; for g in $G; do echo "$g $QD/$v.rad" >> $QD/$v.mix; done
  echo "y4-$v $BASE $QD/$v.mix golden-ver1 corpus/ver1.jsonl" >> $Q
  log "$v converted, queued"
done
echo END >> $Q
./auto/evalq2.sh 0 $Q >> $Y/gate2c-eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $Y/gate2c-eval1.log 2>&1 & wait
S=${RXM_WORK}/evq/scores
for K in 2 3; do
  for v in full-K$K fullr-K$K; do
    [ -f $S/y4-$v.f32 ] || continue
    log "K$K $v vs one-sided: $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y2-1s-K$K.f32 $S/y4-$v.f32)"
  done
  log "K$K full vs sampled (0.025): $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y3-2s32-K$K.f32 $S/y4-full-K$K.f32)"
  log "K$K full vs sampled (0.3):   $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y3-2s32r-K$K.f32 $S/y4-fullr-K$K.f32)"
done
log "gate2c done"
