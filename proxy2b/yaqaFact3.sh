#!/bin/bash
# yaqaFact3.sh -- #33-12 (YAQA08.md amendment 5): mixed / all-diagonal H_out and dose, on base's settings.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 S=/data/yb/snap8192
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact3.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $O/$1.f32 $O/$2.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
# mixed: full H_out, but diagonal for gate / up
[ -f $O/snap8192mix/.done ] || { mkdir -p $O/snap8192mix/hout; for f in $S/hout/*.safetensors; do b=$(basename $f)
  case $b in *mlp.gate_proj*|*mlp.up_proj*) ln -sf $O/snap8192diag/hout/$b $O/snap8192mix/hout/$b;; *) ln -sf $f $O/snap8192mix/hout/$b;; esac; done
  touch $O/snap8192mix/.done; }
for k in 3 2; do
  q k${k}bmix $k --hout $O/snap8192mix/hout --sigma-reg-out 0.01;  pair k${k}base k${k}bmix;  pair k${k}bhout k${k}bmix
  q k${k}bdiag $k --hout $O/snap8192diag/hout --sigma-reg-out 0.01; pair k${k}base k${k}bdiag; pair k${k}bhout k${k}bdiag
done
for n in 1024 2048 4096; do
  q k3bhout$n 3 --hout $O/snap$n/hout --sigma-reg-out 0.01; pair k3base k3bhout$n; pair k3bhout k3bhout$n
done
log FACT3 DONE
