#!/bin/bash
# real107.sh [UNIT...] -- #141-02 (../w107.md): production rxq jobs, 27B layers 2-3 (13 tensors), i8, seed 0, today's
# v3.toml (stamp checked), H_out /data/swap27/hout-bn, --save-wq, K0.125 / K0.5, arms cur (rxq as is) and tab (through
# rxqdrift107.py: LDLQ_DRIFT 0.125 1.348, 0.5 1.184, 1 1.089). Each unit = one rxq run under /root/card0.lock then
# /root/ram.lock shared (queued behind the other card-0 work); finished units (.done) are skipped; stop: touch $W/STOP.
# Then "real": w100's real100.py (unchanged) on every unit's t-files (CPU, niced, ram.lock shared) -> $W/real-ARM-K.jsonl
# Usage (gpubox): setsid nohup bash /data/w107/code/v3/e3/settle/w107/real107.sh > /data/w107/real.out 2>&1 &
set -u
W=/data/w107; CODE=$W/code; HOUT=/data/swap27/hout-bn; S=$CODE/v3/e3/settle
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib PY=/data/venv-exl3/bin/python
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$W/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
ENVQ=(env PYTHONPATH=$CODE/pipeline:$W/exl3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1)
mkdir -p $W/q; cd $W
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/result.txt; }

unit() { # unit ARM K
  local arm=$1 K=$2; local name=$arm-K$K; local d=$W/q/$name/run
  [ -f $d/.done ] && return 0; [ -f $W/STOP ] && { log "STOP before $name"; exit 0; }
  local cmd=(-m rxq); [ $arm = tab ] && cmd=($S/w107/rxqdrift107.py)
  local t=$(date +%s); rm -rf $W/q/$name; mkdir -p $d
  ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
    log "$name: card 0 held after $(( $(date +%s) - t )) s"
    /data/swap27/tools/memguard.sh 3000 nice -n 5 "${ENVQ[@]}" $PY "${cmd[@]}" quant --layers 2-3 -K $K --codebook i8 --seed-offset 0 \
      --ckpt $MODEL --hess $HESS -o $d --device 0 --no-prof --save-wq --hout $HOUT >> $W/q/$name.log 2>&1 )
  local rc=$?
  [ $rc = 0 ] && [ $(ls $d/t-*.safetensors 2>/dev/null | wc -l) = 13 ] && touch $d/.done
  log "$name: rc $rc, $(ls $d/t-*.safetensors 2>/dev/null | wc -l) t-files, $(( $(date +%s) - t )) s wall (incl. lock wait)"; return $rc
}

real() {
  for u in cur-K0.125 tab-K0.125 cur-K0.5 tab-K0.5; do
    [ -f $W/q/$u/run/.done ] || { log "real: $u not done"; continue; }
    [ -f $W/real-$u.done ] && continue
    ( exec 7>/root/ram.lock; flock -s 7; OMP_NUM_THREADS=4 nice -n 10 $PY $S/w100/real100.py $CODE/pipeline $W/q/$u $MODEL $HESS $HOUT $W/real-$u.jsonl ) \
      >> $W/real.log 2>&1 && [ $(wc -l < $W/real-$u.jsonl) = 13 ] && touch $W/real-$u.done && log "real $u: 13 rows" || log "real $u FAILED (real.log)"
  done
}

log "real107 start (code $(cat $CODE/COMMIT), exl3 $(cat $W/exl3/COMMIT), stamp $("${ENVQ[@]}" $PY -m rxq stamp 2>&1 | tail -1))"
ORDER=(${*:-cur:0.125 tab:0.125 cur:0.5 tab:0.5 real})
for u in "${ORDER[@]}"; do
  case $u in
    real) real ;;
    *) IFS=: read arm K <<< "$u"; unit $arm $K || log "$u FAILED (see $W/q/$arm-K$K.log); continuing" ;;
  esac
done
log "real107 done: ${ORDER[*]}"
