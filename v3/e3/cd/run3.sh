#!/bin/bash
# run3.sh -- #77 / #78 ffn_down jobs on cpu1, memory-aware (the container's cgroup is capped at 128 GiB and an
# ffn_down job grows to ~11 GB: run2.sh's 14 at once were OOM-killed). Starts a job only while cgroup memory is below
# MEMGB and fewer than MAX jobs run; skips jobs that are running or already have their result lines.
cd /data/work/e3t; export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
O=/data/work/e3t/cd-cpu1.jsonl; MEMGB=${MEMGB:-95}; MAX=${MAX:-11}
mem() { echo $(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 )); }
have() { [ -f $O ] && python3 -c "import json,sys; n=sum(1 for l in open('$O') if (lambda d: d['layer']==$1 and d['w']=='ffn_down' and d['bits']==$2 and d['method'].endswith('$3'))(json.loads(l))); sys.exit(0 if n else 1)"; }
for l in 10 30 50; do for K in 2 2.5 3 4; do for job in cd cb; do
  if [ $job = cd ]; then args="CD=1 CD_SWEEPS=3 python3 -W ignore trellis2.py $l ffn_down mul1_i8 $K 256 $O"; tag=mul1_i8
  else args="CD=0 python3 -W ignore trellis2.py $l ffn_down gmul1_i8,cubic0.02_i8,cubic0.05_i8 $K 256 $O"; tag=cubic0.05_i8; fi
  pgrep -f "trellis2.py $l ffn_down $( [ $job = cd ] && echo mul1_i8 || echo gmul1_i8 ).* $K 256" > /dev/null && continue
  have $l $K $tag && continue
  while [ $(pgrep -fc "^python3 -W ignore trellis2") -ge $MAX ] || [ $(mem) -ge $MEMGB ]; do sleep 30; done
  echo "$(date -u +%T) start $job $l $K (mem $(mem) GB)"
  (nice -n 10 env $args > logs/$job-$l-ffn_down-$K.log 2>&1 &)
  sleep 90           # let its memory show up before the next check
done; done; done
wait; echo RUN3 DONE
