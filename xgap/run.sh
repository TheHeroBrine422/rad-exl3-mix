#!/bin/bash
# run.sh SCALE [PAR=16] [MAXSEC=900] [EXTRA] -- every source of the registry at SCALE, PAR at a time (restartable: finished sources skip).
# Env XGAP_ROOT (default /data/xgap/v1). Logs: $XGAP_ROOT/logs/<id>.log. Run on colo from /data/xgap/work.
SCALE=$1; PAR=${2:-16}; MAXSEC=${3:-900}; EXTRA=${4:-}   # EXTRA: e.g. --pilot
HERE=$(cd "$(dirname "$0")" && pwd)
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 XGAP_ROOT=${XGAP_ROOT:-/data/xgap/v1}
mkdir -p $XGAP_ROOT/logs
python3 $HERE/list_sources.py | xargs -P $PAR -I{} sh -c "python3 $HERE/build_source.py {} --scale $SCALE --max-seconds $MAXSEC $EXTRA > $XGAP_ROOT/logs/{}.log 2>&1"
echo RUN DONE
