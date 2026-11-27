#!/bin/bash
# smoke236.sh (box2; setsid nohup ... < /dev/null &) -- w236: the 4-card smoke of the V3.1 finals' 2.50 line (plan236, r6 flags), outside the queue.
# Waits until the queue has no running / waiting job (qq.py status), then takes card locks 0-3 (flock -n, all or none; takes / releases in
# /data/q27/w236/cards.log) and runs, each step skipped when its marker exists (restartable):
#   A  smoke25: steps 0 -> MAXA (110) from scratch, the finals line's flags (snapshots on: ckpt-raw-2.5e+05 .. 1e+06 are written)
#   B  smoke25 resumed from its state.pt to MAXA + 10 (the resume path: carried states rebuilt from tokens)
#   C  deep25: smoke25's state moved to plan step DEEP (mkstate233.py) and 20 steps there (--snap-at none): the plan's deepest region
# The trainer is the queue's deployed r6 (QROOT/code, read-only), launched like qrun.sh (one stage process a card, groups 0,1+2,3) with
# PYTORCH_HIP_ALLOC_CONF as w233 measured r6. A sampler logs MemAvailable (GB) + VRAM used of GPUs 0-3 (GiB) every 10 s to OUT/mem.log. Releases the cards at the end or on error.
set -u
W=/data/q27/w236; MAXA=${MAXA:-110}; DEEP=${DEEP:-632}; DEADLINE=${DEADLINE:-$(date -d '07:00 CST' +%s)}
source /data/q27/queue/code/v3/qadq/qenv.sh box2
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/smoke.log; }
clog() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w236 $*" >> $W/cards.log; }
log "smoke236 waiting for an idle queue (deadline $(TZ=America/Chicago date -d @$DEADLINE '+%-I:%M %p CST'))"
while true; do
  [ $(date +%s) -lt $DEADLINE ] || { log "deadline passed without free cards: nothing run"; exit 3; }
  busy=$($PY $QROOT/code/v3/qadq/qq.py status $QROOT | awk '$2 == "running" || $2 == "waiting" || $2 == "ready" || $2 == "stale"' | wc -l)
  if [ "$busy" = 0 ]; then
    fds=(); ok=1
    for c in 0 1 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; if flock -n $l; then fds+=($l); else ok=0; eval "exec $l>&-"; break; fi; done
    [ $ok = 1 ] && break
    for f in "${fds[@]}"; do eval "exec $f>&-"; done
  fi
  sleep 60
done
clog "takes cards 0,1,2,3 (smoke)"; log "cards 0-3 held"; T0=$(date +%s)
smp=
release() { [ -n "$smp" ] && kill $smp 2>/dev/null; clog "releases cards 0,1,2,3 (smoke, held $(( ($(date +%s) - T0) / 60 )) min)"; log "cards released"; }
trap release EXIT
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "MemAvailable ${AV} GB"
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864 PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
SIZE=2.50
FIN=(--lr 2.5e-4 --V 24 --plan231 $W/plan236.json --pipe-procs --split-layers 32 --state-early --evict-fit --pipe-ahead 4 --state-budget-gb 10.5,9.5
     --state-card-gb 6.5,5.5 --card-mem-gb 29.5 --park-pool-gb 12 --pool-strict --a8-int8 --compile-scales --compile-ew --attn-gqa --cap 262144)
run() {   # run OUT flags...: the 4 stage processes as qrun.sh starts them for 0,1+2,3; rc = the first failing rank's
  local OUT=$1; shift; mkdir -p $OUT
  local ARGS=(--model $MODEL --level $IN/lv/$SIZE/trunk --chunks @$CHUNKS --calib $CALIB --pools $IN/pools214.json --pool train214 --genmask $IN/genmask
    --head $IN/lv/$SIZE/head --head-perm $IN/lv/$SIZE/R16.npy --ab $IN/lv/$SIZE/ab.safetensors --budgets 14.8e6 --fixed-eval 20000 --fe-every 12 --monitor 0
    --build-lock $BUILD_LOCK --save-every 12 "$@")
  ( while true; do v=""; for c in 8 7 6 5; do v="$v $(awk '{printf "%.2f", $1/2^30}' /sys/class/drm/card$c/device/mem_info_vram_used)"; done   # box2 GPU 0-3 = drm card8,7,6,5 (unique ids)
      echo "$(date +%T) $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)$v" >> $OUT/mem.log; sleep 10; done ) & smp=$!
  local pids=() r rc=0 left=4 dp p
  for r in 0 1 2 3; do
    local lg=$OUT/stdout.log; [ $r -gt 0 ] && lg=$OUT/stdout-r$r.log
    env ROCR_VISIBLE_DEVICES=$r RANK=$r WORLD_SIZE=4 LOCAL_RANK=0 QAD_STAGES=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29760 nice -n 5 $PY $TRAINER $OUT "${ARGS[@]}" >> $lg 2>&1 < /dev/null &
    pids+=($!)
  done
  while [ $left -gt 0 ]; do
    wait -n -p dp "${pids[@]}"; r=$?; left=$((left - 1))
    if [ $r -ne 0 ] && [ $rc -eq 0 ]; then rc=$r; log "a rank (pid $dp) rc $r: stopping the others"; kill "${pids[@]}" 2>/dev/null; fi
    pids=($(for p in "${pids[@]}"; do [ $p != $dp ] && echo $p; done))
  done
  kill $smp; smp=; return $rc
}
S=$W/smoke25; D=$W/deep25
if [ ! -e $S/.doneA ]; then log "A: smoke25 steps 0 -> $MAXA"; run $S "${FIN[@]}" --max-steps $MAXA; rc=$?; log "A rc $rc"; [ $rc = 0 ] || exit $rc; touch $S/.doneA; fi
if [ ! -e $S/.doneB ]; then log "B: smoke25 resumed -> $((MAXA + 10))"; run $S "${FIN[@]}" --max-steps $((MAXA + 10)); rc=$?; log "B rc $rc"; [ $rc = 0 ] || exit $rc; touch $S/.doneB; fi
if [ ! -e $D/.doneC ]; then
  log "C: deep25 at plan step $DEEP, 20 steps"; mkdir -p $D
  [ -e $D/state.pt ] || $PY $W/code/v3/qad233/mkstate233.py $S/state.pt $DEEP $D/state.pt
  run $D "${FIN[@]}" --snap-at none --max-steps $((DEEP + 20)); rc=$?; log "C rc $rc"; [ $rc = 0 ] || exit $rc; touch $D/.doneC
fi
log "smoke236 done ($(( ($(date +%s) - T0) / 60 )) min on the cards)"
