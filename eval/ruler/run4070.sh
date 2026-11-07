#!/bin/bash
# run4070.sh RUNDIR CONFIG SELECTION-ARGS... -- one rrun.py unit on the RTX 4070 under the shared GPU lock: waits for
# /data/yb/gpu.lock (up to LOCKWAIT s, default 3600), caps VRAM at min(7.2 GB, free - 1 GB), stops new samples after
# BUDGET s (default 780) and is killed at 840 s. Resumable: rerun the same line to continue. Logs to RUNDIR/unit.log.
X="$(dirname "$(readlink -f "$0")")"; OUT=$1 CFG=$2; shift 2
SET=${SET:-$X/sets/ruler-qwen38-v1} TOKJ=${TOKJ:-/data/models/qwen35-08b/tokenizer.json}
mkdir -p "$OUT"
exec 9>/data/yb/gpu.lock
flock -w "${LOCKWAIT:-3600}" 9 || { echo "$(date -u +%T) gpu.lock wait timed out" >> "$OUT/unit.log"; exit 5; }
read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
cap=$(python3 -c "print(round(min(7.2, ($total - $used) / 1024 - 1.0), 2))")
echo "$(date -u +%T) lock taken; VRAM used ${used} MiB of ${total}; cap ${cap} GB" >> "$OUT/unit.log"
CUDA_VISIBLE_DEVICES=0 RULER_VRAM_CAP_GB=$cap OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 nice -n 10 timeout 840 \
  python3 -I "$X/rrun.py" --set "$SET" --config "$CFG" --out "$OUT" --tokenizer "$TOKJ" --budget-s "${BUDGET:-780}" "$@" \
  >> "$OUT/unit.log" 2>&1
rc=$?; echo "$(date -u +%T) rrun rc=$rc" >> "$OUT/unit.log"; exit $rc
