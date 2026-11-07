#!/bin/bash
# rungpubox.sh RUNDIR CONFIG SELECTION-ARGS... -- one rrun.py hold on gpubox card 0: takes /root/card0.lock, then
# /root/conv.lock (no rad-convert pack beside the engine's load) and /root/ram.lock (shared); inside the hold waits up to 60 s for MEMSTART GB of MemAvailable (default 8.5; the card-0
# quantization's RAM is free while we hold the card; the engine's load dips ~6.3 GB) else gives the hold back (exit 4); the hold lasts at most HOLD s
# (default 570: engine start, samples, stop); rrun stops taking samples at BUDGET s (default 480). Never card 1.
# Resumable: rerun the same line to continue (exit 3 = more to do). Logs to RUNDIR/unit.log.
X="$(dirname "$(readlink -f "$0")")"; OUT=$1 CFG=$2; shift 2
SET=${SET:-/data/work/w66/sets/ruler-qwen38-v1} TOKJ=${TOKJ:-/workspace/models/qwen38-27b-bf16/tokenizer.json}
PY=${PY:-/data/venv-exl3/bin/python}
mkdir -p "$OUT"; L="$OUT/unit.log"
mem() { awk '/MemAvailable/ {printf "%.2f", $2 / 1048576}' /proc/meminfo; }
exec 5>/root/card0.lock; flock 5; exec 9>/root/conv.lock; flock 9; exec 7>/root/ram.lock; flock -s 7
t0=$(date +%s)
until awk -v a="$(mem)" -v b="${MEMSTART:-8.5}" 'BEGIN { exit !(a >= b) }'; do
  [ $(( $(date +%s) - t0 )) -gt 60 ] && { echo "$(date -u +%T) MemAvailable $(mem) GB < ${MEMSTART:-8.5}: hold given back" >> "$L"; exit 4; }
  sleep 2; done
echo "$(date -u +%T) card0 + conv + ram locks taken; MemAvailable $(mem) GB" >> "$L"
ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 timeout -k 20 "${HOLD:-570}" \
  "$PY" -I "$X/rrun.py" --set "$SET" --config "$CFG" --out "$OUT" --tokenizer "$TOKJ" --budget-s "${BUDGET:-480}" "$@" >> "$L" 2>&1
rc=$?; echo "$(date -u +%T) rrun rc=$rc; locks released" >> "$L"; exit $rc
