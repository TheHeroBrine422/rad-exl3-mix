#!/bin/bash
# a113.sh [UNIT...] -- w113 #148-01 (../w113.md): real-quant f2 of N / full P16 (P) / P16-TP (T) on L10 in_proj_qkv,
# L10 in_proj_z, L11 / L39 q_proj, L3 / L11 / L39 / L63 v_proj, plus V (nest 2) on the v_proj, K2.5 / K4 x seed offsets
# 0 / 1, gpubox card 0. Code = w101's staged (c) code + exllamav3 fork (stamp 87a7e4ab, checked per unit). Every unit is
# one rxq quant under /root/card0.lock then /root/ram.lock (shared), one unit per hold; finished units (.done) are
# skipped on a rerun; stop: touch $W/STOP. Steps: prep (mkperm113 P / T / V, CPU) -> units -> f2 -> report.
# A113_LOCKDIR=<dir>: take the step locks there (dummy files) because hold113.sh already holds /root's card0 and ram
# (shared) locks for a short batch of units.
# Usage (gpubox): setsid nohup bash /data/w113/code/v3/e3/settle/w113/a113.sh > /data/w113/a/run.out 2>&1 < /dev/null &
set -u
W=/data/w113/a; S=/data/w113/code/v3/e3/settle/w113; W101=/data/w101/c; CODE=$W101/code; EXL3=/data/w101/exl3
OUT=$W/out; PR=$W/perm; HOUT=/data/swap27/hout-bn STAMP=87a7e4ab32f9d5b9 LK=${A113_LOCKDIR:-/root}
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib PY=/data/venv-exl3/bin/python
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$W101/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY -m rxq quant)
P=model.language_model.layers
VK=$P.3.self_attn.v_proj,$P.11.self_attn.v_proj,$P.39.self_attn.v_proj,$P.63.self_attn.v_proj
KEYS=$P.10.linear_attn.in_proj_qkv,$P.10.linear_attn.in_proj_z,$P.11.self_attn.q_proj,$P.39.self_attn.q_proj,$VK
mkdir -p $OUT $PR; cd $W
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/result.txt; }

prep() {
  for a in P T V; do [ -f $PR/$a/.done ] && continue
    local k=$KEYS; [ $a = V ] && k=$VK
    ( exec 7>$LK/ram.lock; flock -s 7; nice -n 10 $PY $S/mkperm113.py $a $MODEL $HOUT $PR/$a ${k//,/ } ) > $W/prep-$a.log 2>&1 \
      || { log "prep $a FAILED (prep-$a.log)"; exit 1; }
    log "prep $a: $(tail -1 $W/prep-$a.log)"; done
}

unit() { # unit ARM K SEED
  local arm=$1 K=$2 s=$3; local name=$arm-K$K-s$s; local d=$OUT/$name
  [ -f $d/.done ] && return 0; [ -f $W/STOP ] && { log "STOP before $name"; exit 0; }
  local ck=$MODEL ho=$HOUT keys=$KEYS n=8; [ $arm != N ] && { ck=$PR/$arm/ckpt; ho=$PR/$arm/hout; }
  [ $arm = V ] && { keys=$VK; n=4; }
  local t=$(date +%s); mkdir -p $d
  ( exec 5>$LK/card0.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7
    log "$name: card 0 held after $(( $(date +%s) - t )) s"
    /data/swap27/tools/memguard.sh 3000 nice -n 5 "${RX[@]}" --keys $keys -K $K --seed-offset $s --ckpt $ck --hess $HESS \
      -o $d --device 0 --no-prof --save-wq --hout $ho >> $d.log 2>&1 )
  local rc=$?
  grep -q "$STAMP" $d.log $d/*.json 2>/dev/null || { log "$name: settings stamp $STAMP not found: STOP"; exit 1; }
  [ $rc = 0 ] && [ $(ls $d/wq/*.safetensors 2>/dev/null | wc -l) = $n ] && touch $d/.done
  log "$name: rc $rc, $(( $(date +%s) - t )) s wall (incl. lock wait)"; return $rc
}

DEF=(prep); for s in 0 1; do for K in 2.5 4; do for a in N P T V; do DEF+=($a:$K:$s); done; done; done; DEF+=(f2 report)
ORDER=(${*:-${DEF[*]}})
for u in "${ORDER[@]}"; do
  case $u in
    prep) prep ;;
    f2) ( exec 7>$LK/ram.lock; flock -s 7; THREADS=4 W113_PIPE=$CODE/pipeline nice -n 10 $PY $S/f2il113.py $OUT $MODEL $HESS $HOUT $PR ) \
          >> $W/f2.log 2>&1 && log "f2: done" || log "f2 FAILED (f2.log)" ;;
    report) $PY $S/report148a.py $OUT > $W/report148a.txt 2>&1 && log "report: $(tail -1 $W/report148a.txt)" || log "report FAILED" ;;
    *) IFS=: read arm K s <<< "$u"; unit $arm $K $s || log "$u FAILED (see $OUT/$arm-K$K-s$s.log); continuing" ;;
  esac
done
log "a113 done: ${ORDER[*]}"
