#!/bin/bash
# r100fill.sh JOBFILE MAX -- run r100.sh-style job lines (DRY=1 output) from JOBFILE in order, each started only
# while fewer than MAX trellis4.py jobs run on the machine (whoever started them): fills the slots another r100.sh
# leaves free (used to move cpu1's unstarted part-a jobs to cpu2, with the output path rewritten). nice 10, one
# thread a job, run from r100.sh's code dir. Env MEMGB: also wait while the container's anon memory is >= MEMGB GiB
# (ffn_down jobs hold ~9.8 GB, more while loading); GAP: seconds after each start (default 5; ~120 for ffn_down).
cd "$(dirname "$(readlink -f "$0")")"
export E3_FACTORS=0.5,0.7,0.85,1.0,1.15,1.3,1.5 E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CD=0
mkdir -p logs
while read -r line; do
  while [ $(pgrep -fc '^python3 -W ignore trellis4.py') -ge $2 ] ||
        [ $(( $(grep '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2) >> 30 )) -ge ${MEMGB:-100000} ]; do sleep 20; done
  nohup nice -n 10 bash -c "$line" < /dev/null > /dev/null 2>&1 &
  sleep ${GAP:-5}
done < "$1"
wait
echo FILL DONE $1
