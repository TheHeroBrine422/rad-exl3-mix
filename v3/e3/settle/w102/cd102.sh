#!/bin/bash
# cd102.sh [UNIT...] -- w102 #136-03 (../w102.md): tile CD no-harm at K6 / K1.5 on the real quantizer, gpubox card 0.
# 27B self_attn.q_proj (attn_qg) of layers 3 / 23 / 43 / 63, seed offset 0, v3.toml as staged (the rule commit): arm cd =
# config as is (tile CD on attn_qg), arm off = --set quant.tile_cd.enable=false. Every unit is one rxq quant run under
# /root/card0.lock + /root/ram.lock (shared), so the card-0 queue interleaves between units; finished units (.done)
# are skipped on a rerun; stop: touch $W/STOP. Then f2 (f2cd102.py, CPU niced, ram.lock shared).
# Usage (gpubox): nohup bash /data/w102/code/v3/e3/settle/w102/cd102.sh > /data/w102/run.out 2>&1 &
set -u
W=/data/w102; CODE=$W/code; OUT=$W/out; HOUT=/data/swap27/hout-bn; S=$CODE/v3/e3/settle/w102
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib PY=/data/venv-exl3/bin/python
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$W/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$W/exl3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY -m rxq quant)
P=model.language_model.layers
KEYS=$P.3.self_attn.q_proj,$P.23.self_attn.q_proj,$P.43.self_attn.q_proj,$P.63.self_attn.q_proj
mkdir -p $OUT; cd $W
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/result.txt; }

unit() { # unit ARM K
  local arm=$1 K=$2; local name=$arm-K$K; local d=$OUT/$name
  [ -f $d/.done ] && return 0; [ -f $W/STOP ] && { log "STOP before $name"; exit 0; }
  local extra=(); [ $arm = off ] && extra=(--set quant.tile_cd.enable=false)
  local t=$(date +%s); rm -rf $d; mkdir -p $d
  ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
    log "$name: card 0 held after $(( $(date +%s) - t )) s"
    /data/swap27/tools/memguard.sh 3000 nice -n 5 "${RX[@]}" --keys $KEYS -K $K --seed-offset 0 --ckpt $MODEL --hess $HESS \
      -o $d --device 0 --no-prof --save-wq --hout $HOUT "${extra[@]}" >> $d.log 2>&1 )
  local rc=$?
  [ $rc = 0 ] && [ $(ls $d/wq/*.safetensors 2>/dev/null | wc -l) = 4 ] && touch $d/.done
  log "$name: rc $rc, $(( $(date +%s) - t )) s wall (incl. lock wait)"; return $rc
}

ORDER=(${*:-cd:6 off:6 cd:1.5 off:1.5 f2})
for u in "${ORDER[@]}"; do
  case $u in
    f2) ( exec 7>/root/ram.lock; flock -s 7; THREADS=4 W102_PIPE=$CODE/pipeline nice -n 10 $PY $S/f2cd102.py $OUT $MODEL $HESS $HOUT ) \
          >> $W/f2.log 2>&1 && log "f2: done ($OUT/f2.json)" || log "f2 FAILED (f2.log)" ;;
    *) IFS=: read arm K <<< "$u"; unit $arm $K || log "$u FAILED (see $OUT/$arm-K$K.log); continuing" ;;
  esac
done
log "cd102 done: ${ORDER[*]}"
