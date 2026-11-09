#!/bin/bash
# stampchk92.sh -- (gpubox, card 0, ~5 min) lead note 11-08 5:05 PM CST: the w92 levels carry code stamp 92132b43 (rad-exl3-mix
# 50902fc + fork 6242f7ae); later commits changed only code-stamped files that do not decide trunk codes (w93: rxq/ladder.py;
# w94: fork patch 0009, a device move in the proxy metric). Re-make 3 of fs0's jobs (L02-g0 GDN qkv / z, L03-g0 q / k / v
# with tile CD, L03-g3 down) with the current trees (rad-exl3-mix $REV, fork $FORKREV; staged under /data/w92/codeH,
# /data/w92/exl3H) and compare them with fs0's level tensor for tensor (rxq.pairs same): PAIR OK = the anchors' codes are
# what the freeze code makes (only the stamp string differs). Staging is done from the VM (see the report).
set -u
: ${RXM_WORK:=/root}
A=/data/w92; C=$A/codeH; X=$A/exl3H; PY=/data/venv-exl3/bin/python; T=/data/swap27/tools; OUT=$A/stampchk
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$A/buildH OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$C/pipeline:$X:/data/exllamav3 RXQ_RXM=$C RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
mkdir -p $OUT; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $OUT/result.txt; }
[ -d $OUT/jobs ] || "${RX[@]}" $C/v3/e3/settle/w92/mkjobs92.py 2.5-i8 0 /data/workspace/models/qwen38-27b-bf16 /root/challenge/hessians/calib \
  /data/swap27/hout-bn $OUT/jobs --layers 2-3 > $OUT/mkjobs.log 2>&1 || { log "FAIL mkjobs"; exit 1; }
for n in L02-g0 L03-g0 L03-g3; do
  [ -f $OUT/$n/job.json ] && continue
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    $T/memguard.sh 3000 nice -n 5 "${RX[@]}" -m rxq job $OUT/jobs/$n.json -o $OUT/$n --device 0 ) >> $OUT/$n.log 2>&1 || { log "FAIL job $n"; exit 1; }
done
for n in L02-g0 L03-g0 L03-g3; do
  r=$("${RX[@]}" -m rxq.pairs same $A/lv/fs0/level $OUT/$n | tail -n 1)
  log "$n: code stamp $($PY -c "import json; print(json.load(open('$OUT/$n/job.json'))['code_stamp'])") vs fs0 92132b43...: $r"
done
log "STAMPCHK92 DONE"
