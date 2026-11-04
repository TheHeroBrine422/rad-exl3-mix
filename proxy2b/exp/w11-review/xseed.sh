#!/bin/bash
# xseed.sh -- w11 (lead launches, 4070 GPU, 4 evals ~4 min, no quantization): does #87's sibling-asym correction
# transfer to siblings it was not fitted on? (F - I) of the same tensor built on seed 0 / 1 / 2 siblings has cosine
# only 0.02-0.10 (w11 fsim.py), i.e. it mostly compensates one rounding of the siblings. Mix seed 1's 102 non-writer
# tensors (q/k/v, GDN qkv/z, gate/up = the siblings) with seed 0's 48 writers:
#   xs = seed-0 ASYM writers (fitted to seed-0 siblings) on seed-1 siblings
#   xc = seed-0 PLAIN writers (k<K>base) on seed-1 siblings  (the control: same mix of seeds, no correction)
# Reading: xs/xc ~ 1 -> the gain is realization-specific: F must be rebuilt on the exact shipped siblings (K, seed,
# H_out, scale refits, healing); xs/xc ~ 0.86 (K3) / 0.82 (K2) like matched pairs -> it is systematic and transfers.
# Out: /data/yb/w11-review/xseed.txt (+ wq dirs of symlinks, .f32 score files).
P=/workspace/rad-exl3-mix/proxy2b S=/data/g08-search1 Y=/data/yb W7=/data/yb/w07-seqsib O=${O:-/data/yb/w11-review}
M=/data/models/qwen35-08b; mkdir -p $O; cd /tmp; log() { echo "$(date -u +%T) $*" | tee -a $O/xseed.txt; }
mix() { d=$O/wq-$1; [ $(ls $d/*.safetensors 2>/dev/null | wc -l) = 150 ] && return; rm -rf $d; mkdir -p $d
  for f in $2/*.safetensors; do ln -s $(readlink -f $f) $d/; done
  for key in $(cat $W7/only.txt); do ln -sfn $(readlink -f $3/$key.safetensors) $d/$key.safetensors; done
  [ $(ls $d/*.safetensors | wc -l) = 150 ] || { log "ASSEMBLE FAIL $1"; exit 1; }; }
ev() { [ -f $O/$1.f32 ] || log "$1: $(python3 $P/evalq.py $M $S --wq $O/wq-$1 --out $O/$1.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pr() { log "pair $(basename $1 .f32) -> $(basename $2 .f32): $(python3 $P/pcmp.py $S $1 $2 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
if [ -n "$XSEED_DRY" ]; then                             # assembly check only (CPU): O=/tmp/x XSEED_DRY=1 ./xseed.sh
  for k in 3 2; do mix k${k}xs $Y/q-k${k}base_s1/wq $W7/q-k${k}asym/wq; mix k${k}xc $Y/q-k${k}base_s1/wq $Y/q-k${k}base/wq
    for t in xs xc; do src=$([ $t = xs ] && echo $W7/q-k${k}asym/wq || echo $Y/q-k${k}base/wq); nw=0; nb=0
      for f in $O/wq-k$k$t/*.safetensors; do b=$(basename $f .safetensors); r=$(readlink -f $f)
        if grep -qx "$b" $W7/only.txt; then [ $r = $(readlink -f $src/$b.safetensors) ] && nw=$((nw + 1))
        else [ $r = $(readlink -f $Y/q-k${k}base_s1/wq/$b.safetensors) ] && nb=$((nb + 1)); fi; done
      echo "k$k$t: writers from $(basename $(dirname $src)) $nw/48, others from q-k${k}base_s1 $nb/102"; done; done; exit 0; fi
for k in 3 2; do
  mix k${k}xs $Y/q-k${k}base_s1/wq $W7/q-k${k}asym/wq;  ev k${k}xs
  mix k${k}xc $Y/q-k${k}base_s1/wq $Y/q-k${k}base/wq;   ev k${k}xc
  pr $O/k${k}xc.f32 $O/k${k}xs.f32                       # the transfer test
  pr $Y/k${k}base_s1.f32 $O/k${k}xc.f32                  # seed mixing alone (expected ~1, as #87's rs controls)
  pr $Y/k${k}base_s1.f32 $W7/k${k}asym_s1.f32            # matched reference (#87 seed 1)
done
log "XSEED DONE"
