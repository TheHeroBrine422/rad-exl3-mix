#!/bin/bash
# yaqaFact2.sh -- #33-11 (YAQA08.md amendment 4): H_out at K2, and H_out stacked on base's settings at K3 / K2.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 S=/data/yb/snap8192
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact2.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $O/$1.f32 $O/$2.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
P01="--sigma-reg 0.01 --sigma-reg-in 0.01 --sigma-reg-out 0.01"
q k3bhout 3 --hout $S/hout --sigma-reg-out 0.01
pair k3base k3bhout; pair k3hout k3bhout
NOSH=1 q k2hout 2 $P01 --hout $S/hout
pair k2ldlq k2hout; pair k2base k2hout
q k2bhout 2 --hout $S/hout --sigma-reg-out 0.01
pair k2base k2bhout; pair k2hout k2bhout
NOSH=1 q k2hdiag 2 $P01 --hout $O/snap8192diag/hout
pair k2ldlq k2hdiag; pair k2base k2hdiag
log FACT2 DONE
