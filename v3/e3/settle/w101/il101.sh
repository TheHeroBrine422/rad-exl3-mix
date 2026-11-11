#!/bin/bash
# il101.sh [UNIT...] -- w101 #130-01 (../w101.md): realized f2 of the P16 output-row tile interleave on the real
# quantizer, gpubox card 0. 27B L10 in_proj_qkv / in_proj_z, L11 / L39 q_proj at K2.5 / K4 (i8), seed offsets 0 / 1,
# arms N (natural) and P (P16 rows: mkperm101.py's permuted 4-tensor checkpoint + permuted bn H_out), v3.toml settings
# (staged code = the rule commit). Every unit is one rxq quant run under /root/card0.lock + /root/ram.lock (shared), so
# the card-0 queue interleaves between units; finished units (.done) are skipped on a rerun; stop: touch $W/STOP.
# Steps: prep (mkperm, CPU) -> units -> f2 (f2il101.py, CPU niced, ram.lock shared) -> report (report130a.py).
# Usage (gpubox): nohup bash /data/w101/code/v3/e3/settle/w101/il101.sh > /data/w101/run.out 2>&1 &
set -u
W=/data/w101; CODE=$W/code; OUT=$W/out; PD=$W/p16; HOUT=/data/swap27/hout-bn; S=$CODE/v3/e3/settle/w101
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib PY=/data/venv-exl3/bin/python
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$W/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$W/exl3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY -m rxq quant)
P=model.language_model.layers
KEYS=$P.10.linear_attn.in_proj_qkv,$P.10.linear_attn.in_proj_z,$P.11.self_attn.q_proj,$P.39.self_attn.q_proj
mkdir -p $OUT; cd $W
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/result.txt; }

prep() {
  [ -f $PD/.done ] && return 0
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY $S/mkperm101.py $MODEL $HOUT $PD ${KEYS//,/ } ) > $W/prep.log 2>&1 \
    || { log "prep FAILED (prep.log)"; exit 1; }
  log "prep: $(tail -1 $W/prep.log)"
}

unit() { # unit ARM K SEED
  local arm=$1 K=$2 s=$3; local name=$arm-K$K-s$s; local d=$OUT/$name
  [ -f $d/.done ] && return 0; [ -f $W/STOP ] && { log "STOP before $name"; exit 0; }
  local ck=$MODEL ho=$HOUT; [ $arm = P ] && { ck=$PD/ckpt; ho=$PD/hout; }
  local t=$(date +%s); mkdir -p $d
  ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
    log "$name: card 0 held after $(( $(date +%s) - t )) s"
    /data/swap27/tools/memguard.sh 3000 nice -n 5 "${RX[@]}" --keys $KEYS -K $K --seed-offset $s --ckpt $ck --hess $HESS \
      -o $d --device 0 --no-prof --save-wq --hout $ho >> $d.log 2>&1 )
  local rc=$?
  [ $rc = 0 ] && [ $(ls $d/wq/*.safetensors 2>/dev/null | wc -l) = 4 ] && touch $d/.done
  log "$name: rc $rc, $(( $(date +%s) - t )) s wall (incl. lock wait)"; return $rc
}

ORDER=(${*:-prep N:2.5:0 P:2.5:0 N:4:0 P:4:0 N:2.5:1 P:2.5:1 N:4:1 P:4:1 f2 report})
for u in "${ORDER[@]}"; do
  case $u in
    prep) prep ;;
    f2) ( exec 7>/root/ram.lock; flock -s 7; THREADS=4 W101_PIPE=$CODE/pipeline nice -n 10 $PY $S/f2il101.py $OUT $MODEL $HESS $HOUT $PD ) \
          >> $W/f2.log 2>&1 && log "f2: done" || log "f2 FAILED (f2.log)" ;;
    report) $PY $S/report130a.py $OUT > $W/report130a.txt 2>&1 && log "report: $W/report130a.txt" || log "report FAILED" ;;
    *) IFS=: read arm K s <<< "$u"; unit $arm $K $s || log "$u FAILED (see $OUT/$arm-K$K-s$s.log); continuing" ;;
  esac
done
log "il101 done: ${ORDER[*]}"
