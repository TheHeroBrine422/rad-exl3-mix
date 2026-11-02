#!/bin/bash
# calib1.sh -- the 2B calibration study (CALIBRATION.md), unattended: search golden, quants, evals, comparisons.
# Waits for setup08 (the 0.8B Hessians) to leave the GPU. Out /data/cal2b/, summary /data/cal2b/result.txt.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-2b O=/data/cal2b G=/data/g2b-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
while pgrep -f "hcollect.py|setup08.sh" > /dev/null; do sleep 30; done
if [ ! -f $G/kld.json ]; then
  D=$(POOLS=/workspace/rad-exl3-mix/eval/pools/pools.json python3 /workspace/rad-exl3-mix/mix/calib_slice.py /data/challenge/corpus/calib.jsonl 300000 2048 1 search 2>$O/slice.txt)
  python3 $P/mkgold.py $M /data/challenge/calib-meta $G $D > $O/gold.log 2>&1; log "golden: $(cat $O/slice.txt)"
fi
( cd /tmp/pb && PYTHONPATH=/data/src/exllamav3 TORCH_EXTENSIONS_DIR=/data/torch_ext python3 /workspace/rad-exl3-mix/v3/e3/exl3frac2/qtf2check.py /data/src/exllamav3/exllamav3/exllamav3_ext 1024 > $O/qtf2check.log 2>&1 )
log "qtf2check: $(grep -c '"exact"' $O/qtf2check.log) exact rows, $(grep '"exact"' $O/qtf2check.log | grep -vc '"idx_diff": 0, "val_diff": 0') nonzero; tail: $(tail -1 $O/qtf2check.log | cut -c1-150)"
python3 -c "
for l in range(24):
    p=f'model.language_model.layers.{l}'
    ks=[p+'.self_attn.'+x for x in ('q_proj','k_proj','v_proj','o_proj')] if l%4==3 else [p+'.linear_attn.'+x for x in ('in_proj_qkv','in_proj_z','out_proj')]
    print('\n'.join(ks+[p+'.mlp.'+x for x in ('gate_proj','up_proj','down_proj')]))" > $O/alltrunk.txt
q() { tag=$1; k=$2; cb=$3; shift 3
  [ -f $O/$tag.f32 ] && return
  [ -d $O/q-$tag/wq ] && [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || \
    $P/q2b.sh 2b $k $cb $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  s=$(date +%s); r=$(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1)
  log "$tag: $r"; }
q k3i8 3 i8
q k2i8 2 i8
q k2.5i8 2.5 i8
q k3.5i8 3.5 i8
q k4i8 4 i8
q k3mul1 3 mul1
q k3i8os 3 i8 --out-scales-off $O/alltrunk.txt
NOSH=1 q k2.5i8nosh 2.5 i8
for pr in "k3i8 k3mul1" "k3i8 k3i8os" "k2.5i8 k2.5i8nosh" "k2i8 k2.5i8" "k2.5i8 k3i8" "k3i8 k3.5i8" "k3.5i8 k4i8"; do set -- $pr
  [ -f $O/$1.f32 ] && [ -f $O/$2.f32 ] && log "pair $1 -> $2: $(python3 $P/pcmp.py $G $O/$1.f32 $O/$2.f32)"; done
log CALIB1 DONE
