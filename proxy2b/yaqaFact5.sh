#!/bin/bash
# yaqaFact5.sh -- #33-15 (YAQA08.md amendment 8): x2 / bf16 collected diagonals (1K seqs) judged by KL vs the fp32 one.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact5.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $O/$1.f32 $O/$2.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
cd /tmp
[ -f $O/prec1k/bf16/hout/model.language_model.layers.23.mlp.down_proj.safetensors ] || \
  python3 $P/prectest.py $M /data/xcorp/v1/meta $O/prec1k.jsonl --nseq 1024 --modes x2,bf16 --save $O/prec1k > /data/logs/prec1k.log 2>&1
log "collected: $(grep -E 's/seq' /data/logs/prec1k.log | tr '\n' ' ')"
for m in x2 bf16; do for k in 3 2; do
  q k${k}bd$m $k --hout $O/prec1k/$m/hout --sigma-reg-out 0.01; pair k${k}bdiag1024 k${k}bd$m; pair k${k}base k${k}bd$m
done; done
log FACT5 DONE
