#!/bin/bash
# check.sh -- w79 confounder check for d1105 k45lr's K5 lr 3e-5 arm (golden 1.144 while held-out -0.8%): rescore the
# stack and the folded arm (with and without its folded norms) on g08-search1. Holds /data/yb/gpu.lock. Idempotent.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b S=/data/g08-search1 D=/data/yb/d1105 O=/data/yb/w79/check
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; cd /tmp; exec 9>/data/yb/gpu.lock; flock 9
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
ev() { [ -f $O/$1.f32 ] && return; log "$1: $(python3 $P/evalq.py $3 $S --wq $2 --out $O/$1.f32 2>/dev/null | tail -1)"; }
ev stack5 $D/q-k5edyswjasym_r0/wqall $M
ev lr3e-5 $D/h-k5edyswjasym_r0_lr3e-5/wq $D/h-k5edyswjasym_r0_lr3e-5/model
ev lr3e-5_nonorm $D/h-k5edyswjasym_r0_lr3e-5/wq $M
ev lr1e-4_nonorm $D/h-k5edyswjasym_r0_lr1e-4/wq $M
for t in lr3e-5 lr3e-5_nonorm lr1e-4_nonorm; do log "pair stack5 -> $t: $(python3 $P/pcmp.py $S $O/stack5.f32 $O/$t.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; done
log "CHECK DONE"
