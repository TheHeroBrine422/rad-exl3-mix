#!/bin/bash
# eq215.sh NAME [qad181.py args]  (w215, iceman card 0) -- the trainer-speed equivalence run: 20 optimizer steps of
# qad181.py (V 6 x Q 1024, seed 0, fixed plan, qad-pool calib docs, gen rows) on a K2.5 int8 student, card 0 only
# (03:00.0; card 1 serves the user's model). Output /data/w215/runs/NAME (train.jsonl per-step losses, state.pt = the
# final scales; CODE=DIR runs another code copy). Runs through prof215.py (PROF215_STEP=k profiles step k); extra args go to qad181.py. Compare runs with eq215.py.
set -u
NAME=$1; shift
W=/data/w215; CODE=${CODE:-$W/code}; PY=/data/venv-exl3/bin/python
export QAD_PIPELINE=$CODE/pipeline TRITON_DEFAULT_FP_FUSION=1 OMP_NUM_THREADS=6
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
OUT=$W/runs/$NAME; mkdir -p $OUT
exec {fd}>/root/card0.lock; flock -n $fd || { echo "card 0 busy"; exit 75; }
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AV" -ge 30 ] || { echo "MemAvailable ${AV} GB < 30"; exit 76; }
ROCR_VISIBLE_DEVICES=0 nice -n 5 $PY $CODE/v3/qad181/prof215.py $OUT --model /data/workspace/models/qwen38-27b-bf16 \
  --level /data/v3ladder2/K2.5 --chunks , --calib /data/challenge/golden/calib --pools $CODE/eval/pools/pools-v3.json \
  --pool qad --V 6 --Q 1024 --budgets 2e5 --max-steps 20 --monitor 0 --save-every 1000 --seed 0 "$@" >> $OUT/stdout.log 2>&1
echo "rc $?" >> $OUT/stdout.log
