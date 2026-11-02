#!/bin/bash
# yaqaBrun.sh -- YAQA Sketch B dose-response on the 0.8B (YAQA08.md, amended 2026-11-02 for Sketch B), unattended:
# collect to 8K sequences (snapshots 1K/2K/4K/8K) -> quants + evals for those -> resume to 16K -> its quants + evals.
# Arms at K2 and K3 (i8, the 0.8B search golden):
#   base  our tuned one-sided quant (scaled Hessian, damping 0.025 / input 0.005, calib input Hessians)
#   ldlq  the paper's one-sided baseline: no scaled Hessian, damping 0.01, calib input Hessians
#   ybN   YAQA-B after N sequences: per-linear H_in and H_out from Sketch B, damping 0.01 both sides, no scaled Hessian
# Out /data/yb/, summary /data/yb/result.txt.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
P01="--sigma-reg 0.01 --sigma-reg-in 0.01 --sigma-reg-out 0.01"
evals() { for k in 3 2; do
    q k${k}base $k
    NOSH=1 q k${k}ldlq $k $P01
    for n in "$@"; do NOSH=1 q k${k}yb$n $k $P01 --hin $O/snap$n/hin --hout $O/snap$n/hout; done
    for n in "$@"; do for ref in ldlq base; do
      [ -f $O/k${k}yb$n.f32 ] && log "pair k${k}$ref -> k${k}yb$n: $(python3 $P/pcmp.py $G $O/k${k}$ref.f32 $O/k${k}yb$n.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; done; done
  done; }
# early read (user, 2026-11-02): 1K snapshot first, its arms, then resume (bit-exact) to 8K
[ -f $O/snap1024/info.json ] || python3 $P/yaqaB.py $M /data/xcorp/v1/meta $O 1024 --snap 1024 > $O/collect0.log 2>&1
[ -f $O/snap1024/info.json ] || { log "COLLECT 1K FAIL"; exit 1; }
evals 1024
[ -f $O/snap8192/info.json ] || python3 $P/yaqaB.py $M /data/xcorp/v1/meta $O 8192 --snap 2048,4096,8192 --resume 1024 > $O/collect1.log 2>&1
log "collect to 8K: $(grep -E 'YAQAB DONE|Error' $O/collect1.log | tail -1)"
[ -f $O/snap8192/info.json ] || { log "COLLECT FAIL"; exit 1; }
evals 2048 4096 8192
[ -f $O/snap16384/info.json ] || python3 $P/yaqaB.py $M /data/xcorp/v1/meta $O 16384 --snap 16384 --resume 8192 > $O/collect2.log 2>&1
log "collect to 16K: $(grep -E 'YAQAB DONE|Error' $O/collect2.log | tail -1)"
evals 16384
log YAQAB RUN DONE
