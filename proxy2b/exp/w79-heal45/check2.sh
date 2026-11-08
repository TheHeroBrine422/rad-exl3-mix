#!/bin/bash
# check2.sh -- w79: the K5 / K4 k45lr arms through heal.py's student in memory (evalm.py) vs their folds. gpu.lock held.
P=/workspace/rad-exl3-mix/proxy2b W=$P/exp/w79-heal45 S=/data/g08-search1 D=/data/yb/d1105 O=/data/yb/w79/check
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; cd /tmp; exec 9>/data/yb/gpu.lock; flock 9
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
em() { [ -f $O/$1.f32 ] && return; log "$1: $(python3 $W/evalm.py $2 $3 $S $O/$1.f32 2>&1 | tail -1)"; }
pr() { log "pair $1 -> $2: $(python3 $P/pcmp.py $S $O/$1.f32 $O/$2.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
em m-stack5 - $D/q-k5edyswjasym_r0/wqall
em m-lr3e-5 $D/h-k5edyswjasym_r0_lr3e-5/params.pt $D/q-k5edyswjasym_r0/wqall
em m-lr1e-4 $D/h-k5edyswjasym_r0_lr1e-4/params.pt $D/q-k5edyswjasym_r0/wqall
pr stack5 m-stack5; pr m-stack5 m-lr3e-5; pr m-stack5 m-lr1e-4
log "CHECK2 DONE"
