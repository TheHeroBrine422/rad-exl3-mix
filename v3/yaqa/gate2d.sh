#!/bin/bash
# gate2d.sh -- diagonal-only output weighting (mining merge D1): one-sided LDLQ, output channel scales from diag(H_out)
# (exllamav3's sensitivity_out_scales, = convert.py --out_scales yaqa; ourq.py --hout-diag), blocks 22+23 at K2/K3,
# with the diagonal from gate 2b's sampled fp32 H_out (hout32: rows for all 496 linears exist, so a whole-model
# version would cost nothing more) and from gate 2c's full-token H_out. Swapped into V2 6.00, golden-ver1, paired vs
# gate 2's one-sided quants (same quantizer, seeds, slice). After gate2c.sh.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa QD=/data/yaqa/qdiag
BASE=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad
PY=/data/venv-exl3/bin/python
log() { echo "$(date -u +%H:%M) $*" | tee -a $Y/gate2d.log; }
until grep -q "gate2c done" $Y/gate2c.log 2>/dev/null || ! pgrep -f "^/bin/bash /data/yaqa/gate2c.sh" > /dev/null; do sleep 60; done
cd ${RXM_WORK}
export EXL3_QT_OPTIMIZED=1
quant() {  # CARD then triples "VARIANT K HOUTDIR"
  local c=$1; shift
  while [ $# -gt 0 ]; do
    local v=$1 K=$2 hd=$3; shift 3
    [ -f $QD/$v/q-L23.safetensors ] && continue
    [ -f $hd/model.language_model.layers.23.mlp.down_proj.safetensors ] || { log "$v: no H_out in $hd"; continue; }
    ( exec 9>${RXM_WORK}/card$c.lock; flock 9
      $PY -I ourq/ourq.py -K $K -o $QD/$v --layers 22-23 --device 0 --hout-diag $hd 2>&1 \
        | grep -v -e Warning -e "cpu = _conv" ) > $QD/$v.log 2>&1 && grep -q '^done' $QD/$v.log || { log "$v quantize FAIL"; continue; }
    log "$v quantized ($(grep -c '"hout_diag_scales": true' $QD/$v/proxy.jsonl) tensors with diagonal scales)"
  done
}
mkdir -p $QD
ROCR_VISIBLE_DEVICES=0 quant 0 dsamp-K2 2 $Y/hout32 dfull-K3 3 $Y/full32/hout &
ROCR_VISIBLE_DEVICES=1 quant 1 dsamp-K3 3 $Y/hout32 dfull-K2 2 $Y/full32/hout &
wait
T22="ffn_down ffn_gate_up ssm_out ssm_qkv ssm_z"; T23="attn_k attn_output attn_qg attn_v ffn_down ffn_gate_up"
G=""; for t in $T22; do G="$G blk.22.$t.weight"; done; for t in $T23; do G="$G blk.23.$t.weight"; done
Q=$Y/gate2d.q; : > $Q
for v in dsamp-K2 dfull-K2 dsamp-K3 dfull-K3; do
  [ -f $QD/$v/q-L23.safetensors ] || continue
  ./ourq/ourconv.sh $QD/$v.rad sc4.00-e4 $QD/$v $G >> $Y/gate2d.log 2>&1 || { log "$v conv FAIL"; continue; }
  : > $QD/$v.mix; for g in $G; do echo "$g $QD/$v.rad" >> $QD/$v.mix; done
  echo "y5-$v $BASE $QD/$v.mix golden-ver1 corpus/ver1.jsonl" >> $Q
  log "$v converted, queued"
done
echo END >> $Q
./auto/evalq2.sh 0 $Q >> $Y/gate2d-eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $Y/gate2d-eval1.log 2>&1 & wait
S=${RXM_WORK}/evq/scores
for K in 2 3; do
  for v in dsamp-K$K dfull-K$K; do
    [ -f $S/y5-$v.f32 ] || continue
    log "K$K $v vs one-sided: $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y2-1s-K$K.f32 $S/y5-$v.f32)"
  done
done
log "gate2d done"
