#!/bin/bash
# bench105.sh [ARM...] -- #143 (w105) throughput arms on gpubox card 0: 20 K2.5 i8 group jobs (layers 8-12: GDN + MLP,
# L11 attention with tile CD, four down_proj), cold page cache, settings = the staged pipeline/config/v3.toml.
# p1 = one `python -m rxq job` process a job (w92's way, the reference); w1 / w2 / w3 = persistent workers N = 1 / 2 / 3;
# p2 = processes N = 2. Each arm takes card0.lock (+ ram.lock shared) for itself only; finished arms are skipped.
# Staged tree: $CODE (pipeline, store, mix, v3/e3/{exl3frac,exl3frac2,exl3cb,settle/w92,settle/w105}), fork in $A/exl3.
set -u
A=${A:-/data/w105}; export CODE=$A/code ROOT=$A
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$A/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1 \
       PYTHONPATH=$CODE/pipeline:$A/exl3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1
PY=/data/venv-exl3/bin/python B=$CODE/v3/e3/settle/w105/bench105.py
ARMS=${*:-p1 w1 w2 w3 p2}
for arm in $ARMS; do
  [ -f $A/HOLD ] && { echo "$(date -u +%H:%M:%S) HOLD: stop before $arm" | tee -a $A/result.txt; exit 0; }
  case $arm in
    p1) $PY $B p1 --mode proc -n 1;;
    p2) $PY $B p2 --mode proc -n 2 --ref p1;;
    w1) $PY $B w1 --mode persist -n 1 --ref p1;;
    w2) $PY $B w2 --mode persist -n 2 --ref p1;;
    w3) $PY $B w3 --mode persist -n 3 --ref p1;;
    *) echo "unknown arm $arm"; exit 1;;
  esac > $A/logs/bench-$arm.log 2>&1 || { echo "$(date -u +%H:%M:%S) $arm FAILED ($A/logs/bench-$arm.log)" | tee -a $A/result.txt; exit 1; }
done
echo "$(date -u +%H:%M:%S) BENCH DONE ($ARMS)" | tee -a $A/result.txt
