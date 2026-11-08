#!/bin/bash
# chain1.sh -- card 1 overnight (2026-10-31), after the both-card YAQA collection:
#  1. trellis embedding, real: K2/2.5/3/3.5/4 (embgpu.py, one scale per row) vs V2's Lloyd e2/e3/e4/e5, each swapped
#     into V2 6.00 (largest embedding share of KLD), scored on golden-fit3 (fit pool: a measurement) + golden-ml
#     (multilingual guard; rare tokens are where a coarse embedding hurts).
#  2. ours vs turboderp at K6 (DESIGN §1d): base ours-K5; every tensor of the even layers swapped to ours-K6 / his
#     plain6.00 (uniformly K6) / ours-K6 requantized with sigma_reg 0.005 (the over-damping hypothesis); golden-fit3.
# Evals on card 1 only (evalq2.sh 1). Log /data/v3levels/chain1.log.
: ${RXM_WORK:=/root}; export RXM_WORK
W=/data/v3levels PY=/data/venv-exl3/bin/python L=/data/v3levels/chain1.log F=/root/models/final
mkdir -p $W/emb; cd $RXM_WORK
log() { echo "$(date -u +%H:%M) $*" | tee -a $L; }
until grep -q "^DONE" /data/yaqa/full64.log 2>/dev/null || ! pgrep -f "^/bin/bash ./gate2e.sh" > /dev/null; do sleep 60; done
export ROCR_VISIBLE_DEVICES=1 EXL3_QT_OPTIMIZED=1
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build EXL3FRAC_SRC=/data/exl3frac/src
# ---- 1. embedding
( exec 9>$RXM_WORK/card1.lock 7>$RXM_WORK/ram.lock; flock 9; flock -s 7
  $PY -I /data/exl3frac/e3/embgpu.py $F/q38-27b-bf16.rad $W/emb 2,2.5,3,3.5,4 ) >> $W/emb/embgpu.log 2>&1
log "embedding donors: $(ls $W/emb/*.done 2>/dev/null | wc -l)/5; $(grep -h '^{' $W/emb/embgpu.log | tr '\n' ' ')"
B6=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad Q=$W/chain1.q; : > $Q
for G in "golden-fit3 corpus/fit3.jsonl" "golden-ml corpus/ml.jsonl"; do
  set -- $G
  echo "v3e-base6-$1 $B6 - $1 $2" >> $Q
  for e in "e2 $F/q38-27b-sc4.00-e2.rad" "e3 /data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad" "e4 $F/q38-27b-sc4.00-e4.rad" "e5 $F/q38-27b-sc5.00-e5.rad"; do
    set -- $e; echo "token_embd.weight $2" > $W/emb/l$1.mix; set -- $G
    echo "v3e-l${e%% *}-$1 $B6 $W/emb/l${e%% *}.mix $1 $2" >> $Q
  done
  for K in 2 2.5 3 3.5 4; do
    [ -f $W/emb/emb-K$K.rad.done ] || continue
    echo "token_embd.weight $W/emb/emb-K$K.rad" > $W/emb/t$K.mix
    echo "v3e-t$K-$1 $B6 $W/emb/t$K.mix $1 $2" >> $Q
  done
done
# ---- 2. K6 swaps: requant ours-K6 at sigma_reg 0.005 (trunk), then the even-layer mixes
O=$W/ours-K6-d005
if [ ! -f $O.rad ]; then
  ( exec 9>$RXM_WORK/card1.lock; flock 9
    $PY -I ourq/ourq.py -K 6 -o $O --device 0 --sigma-reg 0.005 2>&1 | grep -v -e Warning -e "cpu = _conv" ) > $O.log
  if [ -f $O/q-L63.safetensors ]; then
    ALL='blk.*.attn_qg.weight blk.*.attn_k.weight blk.*.attn_v.weight blk.*.attn_output.weight blk.*.ffn_gate_up.weight blk.*.ffn_down.weight blk.*.ssm_qkv.weight blk.*.ssm_z.weight blk.*.ssm_out.weight'
    ./ourq/ourconv.sh $O.rad sc4.00-e4 $O $ALL >> $L 2>&1 && log "ours-K6 sigma 0.005 built" || log "ours-K6-d005 conv FAIL"
  else log "ours-K6-d005 quantize FAIL"; fi
fi
B5=/data/ours/ours-K5.rad
$PY -I - $B5 > $W/even.txt <<'EOF'
import sqlite3, sys
d = sqlite3.connect("/data/store/index.sqlite")
for (n,) in d.execute("select distinct v.name from locs l join variants v on v.vid=l.vid where l.path=? and v.scheme='exl3' and v.name like 'blk.%'", (sys.argv[1],)):
    if int(n.split(".")[1]) % 2 == 0: print(n)
EOF
for a in "o6 /data/ours/ours-K6.rad" "td6 $F/q38-27b-plain6.00-e6.rad" "o6d $O.rad"; do
  set -- $a; [ -f $2 ] || continue
  awk -v d=$2 '{print $1, d}' $W/even.txt > $W/sw-$1.mix
  echo "v3s-$1-fit3 $B5 $W/sw-$1.mix golden-fit3 corpus/fit3.jsonl" >> $Q
done
echo "v3s-base5-fit3 $B5 - golden-fit3 corpus/fit3.jsonl" >> $Q
echo END >> $Q
log "queued $(grep -vc END $Q) evals (even layers: $(wc -l < $W/even.txt) tensors)"
./auto/evalq2.sh 1 $Q >> $W/chain1-eval.log 2>&1
S=$RXM_WORK/evq/scores
for G in golden-fit3 golden-ml; do
  for t in le2 le3 le4 le5 t2 t2.5 t3 t3.5 t4; do
    [ -f $S/v3e-$t-$G.f32 ] && log "emb $t $G vs base6: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/$G $S/v3e-base6-$G.f32 $S/v3e-$t-$G.f32)"
  done
done
for t in o6 td6 o6d; do
  [ -f $S/v3s-$t-fit3.f32 ] && log "swap $t vs base5: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-fit3 $S/v3s-base5-fit3.f32 $S/v3s-$t-fit3.f32)"
done
[ -f $S/v3s-td6-fit3.f32 ] && log "swap td6 vs o6: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-fit3 $S/v3s-o6-fit3.f32 $S/v3s-td6-fit3.f32)"
[ -f $S/v3s-o6d-fit3.f32 ] && log "swap o6d vs o6: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-fit3 $S/v3s-o6-fit3.f32 $S/v3s-o6d-fit3.f32)"
rm -f $W/emb/emb-K*.rad                                    # scratch (54 GB reflink copies, 2.5 GB unshared each)
log "chain1 done"
