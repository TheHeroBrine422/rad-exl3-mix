#!/bin/bash
# gate2e-blend.sh -- ONLY if gate 2e (100 seqs) came out MARGINAL (rule fixed 2026-11-01 before the verdict): blends
# alpha 0.25 / 0.5 of the gradient-weighted H_in with the plain one (hblend.py), K2 + K3, blocks 22/23, swapped into
# V2 6.00, golden-ver1, paired vs one-sided (y2-1s-K*), as overnight.sh. ~40 min on both cards. Needs the user's OK.
# PASS (-> propose the 400-sequence collection): one alpha wins at both K with z <= -2. Else drop H_in.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa F=/data/yaqa/full64 QD=/data/yaqa/qe PY=/data/venv-exl3/bin/python
BASE=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad
log() { echo "$(date -u +%H:%M) $*" | tee -a $Y/gate2e-blend.log; }
cd $RXM_WORK
for al in 0.25 0.5; do
  [ -d $F/hblend-$al ] || nice $PY -I $Y/hblend.py $F/hin-snap100 $RXM_WORK/challenge/hessians/calib $F/hblend-$al $al >> $Y/gate2e-blend.log 2>&1
done
export EXL3_QT_OPTIMIZED=1
q() {  # CARD VARIANT K ALPHA
  [ -f $QD/$2/q-L23.safetensors ] && return
  ( exec 9>$RXM_WORK/card$1.lock; flock 9
    ROCR_VISIBLE_DEVICES=$1 $PY -I ourq/ourq.py -K $3 -o $QD/$2 --layers 22-23 --device 0 --hin $F/hblend-$4 2>&1 \
      | grep -v -e Warning -e "cpu = _conv" ) > $QD/$2.log 2>&1
  grep -q '^done' $QD/$2.log && log "$2 quantized" || log "$2 quantize FAIL"
}
( q 0 hb25-K2 2 0.25; q 0 hb50-K3 3 0.5 ) & ( q 1 hb25-K3 3 0.25; q 1 hb50-K2 2 0.5 ) & wait
T22="ffn_down ffn_gate_up ssm_out ssm_qkv ssm_z"; T23="attn_k attn_output attn_qg attn_v ffn_down ffn_gate_up"
G=""; for t in $T22; do G="$G blk.22.$t.weight"; done; for t in $T23; do G="$G blk.23.$t.weight"; done
Q=$Y/gate2e-blend.q; : > $Q; V=""
for v in hb25-K2 hb25-K3 hb50-K2 hb50-K3; do
  [ -f $QD/$v/q-L23.safetensors ] || continue
  ./ourq/ourconv.sh $QD/$v.rad sc4.00-e4 $QD/$v $G >> $Y/gate2e-blend.log 2>&1 || { log "$v conv FAIL"; continue; }
  : > $QD/$v.mix; for g in $G; do echo "$g $QD/$v.rad" >> $QD/$v.mix; done
  echo "y7-$v $BASE $QD/$v.mix golden-ver1 corpus/ver1.jsonl" >> $Q; V="$V $v"
done
echo END >> $Q
./auto/evalq2.sh 0 $Q >> $Y/gate2e-blend-eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $Y/gate2e-blend-eval1.log 2>&1 & wait
S=$RXM_WORK/evq/scores
for v in $V; do
  K=${v##*-K}; [ -f $S/y7-$v.f32 ] || { log "$v: no score"; continue; }
  log "$v vs one-sided: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-ver1 $S/y2-1s-K$K.f32 $S/y7-$v.f32)"
done
rm -f $QD/hb*.rad
log "gate2e-blend done"
