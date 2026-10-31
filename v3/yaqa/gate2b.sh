#!/bin/bash
# gate2b.sh -- YAQA gate 2 rerun with an fp32 backward (noise.sh: the bf16 backward's rows are wrong for a minority of
# positions, growing toward early layers; H_out diagonal cosine vs fp32 ~0.5 at blocks 22/23).
# 200 qad sequences collected with YAQA_FP32=1 (same windows / positions as the bf16 pilot), H_out folded for blocks
# 22+23, two-sided K2/K3 at the default output regularisation (0.025) and at 0.3, swapped into the V2 6.00 release,
# scored on golden-ver1. The base and one-sided runs are gate2.sh's (same quantizer, seeds and slice), not repeated.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa ROWS=/data/yaqa/rows32 HO=/data/yaqa/hout32 QD=/data/yaqa/q32
BASE=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad
PY=/data/venv-exl3/bin/python
log() { echo "$(date -u +%H:%M) $*" | tee -a $Y/gate2b.log; }
cd ${RXM_WORK}
if ! grep -q "^DONE" $Y/pilot32.log 2>/dev/null; then
  ( exec 8>${RXM_WORK}/card0.lock 9>${RXM_WORK}/card1.lock; flock 8; flock 9
    cd $Y && YAQA_FP32=1 YAQA_START=0 $PY -I collect.py $ROWS 200 2048 32 ) > $Y/pilot32.log 2>&1
  grep -q "^DONE" $Y/pilot32.log || { log "fp32 pilot FAIL"; exit 1; }
fi
log "fp32 pilot done: $(ls $ROWS/.done | wc -l) sequences"
[ -f $HO/model.language_model.layers.23.self_attn.v_proj.safetensors ] || \
  FOLD_THREADS=14 $PY -I $Y/fold.py $ROWS $HO 22,23 >> $Y/gate2b.log 2>&1 || { log "fold FAIL"; exit 1; }
log "folded $(ls $HO | wc -l) H_out"
export EXL3_QT_OPTIMIZED=1
quant() {  # CARD then pairs "VARIANT K SIGMA_OUT" (each card gets one K2 and one K3 run: two-sided K3 is the slow one)
  local c=$1; shift
  while [ $# -gt 0 ]; do
    local v=$1 K=$2 so=$3; shift 3
    ( exec 9>${RXM_WORK}/card$c.lock; flock 9
      $PY -I ourq/ourq.py -K $K -o $QD/$v --layers 22-23 --device 0 --hout $HO --sigma-reg-out $so 2>&1 \
        | grep -v -e Warning -e "cpu = _conv" ) > $QD/$v.log 2>&1 && grep -q '^done' $QD/$v.log || { log "$v quantize FAIL"; continue; }
    log "$v quantized"
  done
}
mkdir -p $QD
ROCR_VISIBLE_DEVICES=0 quant 0 2s32-K2 2 0.025 2s32r-K3 3 0.3 &
ROCR_VISIBLE_DEVICES=1 quant 1 2s32-K3 3 0.025 2s32r-K2 2 0.3 &
wait
T22="ffn_down ffn_gate_up ssm_out ssm_qkv ssm_z"; T23="attn_k attn_output attn_qg attn_v ffn_down ffn_gate_up"
G=""; for t in $T22; do G="$G blk.22.$t.weight"; done; for t in $T23; do G="$G blk.23.$t.weight"; done
Q=$Y/gate2b.q; : > $Q
for v in 2s32-K2 2s32r-K2 2s32-K3 2s32r-K3; do
  [ -f $QD/$v/q-L23.safetensors ] || continue
  ./ourq/ourconv.sh $QD/$v.rad sc4.00-e4 $QD/$v $G >> $Y/gate2b.log 2>&1 || { log "$v conv FAIL"; continue; }
  : > $QD/$v.mix; for g in $G; do echo "$g $QD/$v.rad" >> $QD/$v.mix; done
  echo "y3-$v $BASE $QD/$v.mix golden-ver1 corpus/ver1.jsonl" >> $Q
  log "$v converted, queued"
done
echo END >> $Q
./auto/evalq2.sh 0 $Q >> $Y/gate2b-eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $Y/gate2b-eval1.log 2>&1 & wait
S=${RXM_WORK}/evq/scores
for K in 2 3; do
  for v in 2s32-K$K 2s32r-K$K; do
    [ -f $S/y3-$v.f32 ] || continue
    log "K$K $v vs one-sided: $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y2-1s-K$K.f32 $S/y3-$v.f32)"
    log "K$K $v vs base:      $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y2-base.f32 $S/y3-$v.f32)"
  done
  log "K$K one-sided vs base: $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y2-base.f32 $S/y2-1s-K$K.f32)"
done
log "gate2b done"
