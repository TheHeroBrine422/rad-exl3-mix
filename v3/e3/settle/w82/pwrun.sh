#!/bin/bash
# w82 #118 2(a): pw82.py over the six swj arms, 3 at a time, nice 19, one thread each, CPU only. Restartable (finished
# layers skipped by file). Log /data/w82/pw/run.log; end line "PW82 DONE".
cd /data/w82
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=
run() {  # ARM LEVEL
  nice -n 19 /data/venv-exl3/bin/python -W ignore code/pw82.py $1 $2 /data/swj27/fit/$1 /data/swap27/hout-bn pw > pw/$1.log 2>&1 \
    && echo "$(date -u +%T) done $1" >> pw/run.log || echo "$(date -u +%T) FAIL $1 rc=$?" >> pw/run.log
}
export -f run
echo "$(date -u +%T) start" >> pw/run.log
printf '%s\n' "base /data/v3ladder2/K2.5" "base4 /data/swap27/q-base4/K4" "bn /data/swap27/q-bn/K2.5" \
  "s1 /data/swap27/q-s1/K2.5" "bn4 /data/swap27/q-bn4/K4" "bns1 /data/swap27/q-bns1/K2.5" | xargs -P 3 -L 1 bash -c 'run "$@"' _
echo "$(date -u +%T) PW82 DONE" >> pw/run.log
