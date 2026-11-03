#!/bin/bash
# run5.sh -- #79 on Thor: CD (3 sweeps) on the cubic0.02_i8 codebook, K2.5 / K3, 16 tensors, rows seed 0.
# One thread per job; memory-aware as run3.sh; skips jobs whose log says done.
cd /data/work/e3t; export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib E3_ROWSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
O=/data/work/e3t/cd-thor-stack.jsonl; MEMGB=${MEMGB:-110}; MAX=${MAX:-16}; mkdir -p logs
mem() { echo $(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 )); }
T="10:ffn_down 30:ffn_down 50:ffn_down 10:ffn_gate_up 30:ffn_gate_up 50:ffn_gate_up 10:ssm_qkv 10:ssm_out 30:ssm_qkv 30:ssm_out 50:ssm_qkv 50:ssm_out 19:attn_qg 19:attn_output 43:attn_qg 43:attn_output"
for t in $T; do l=${t%%:*}; w=${t#*:}; for job in cd; do
  if [ $job = cd ]; then args="CD=1 CD_SWEEPS=3 python3 -W ignore trellis2.py $l $w cubic0.02_i8 2.5,3 256 $O"
  else args="CD=0 python3 -W ignore trellis2.py $l $w cubic0.02_i8 2.5,3 256 $O"; fi
  lg=logs/stack-$job-$l-$w.log; grep -q '"bits": 3.0' $lg 2>/dev/null && continue
  while [ $(pgrep -fc "^python3 -W ignore trellis2") -ge $MAX ] || [ $(mem) -ge $MEMGB ]; do sleep 30; done
  echo "$(date -u +%T) start $job $l $w (mem $(mem) GB)"
  (nice -n 10 env $args > $lg 2>&1 &)
  case $w in ffn_*) sleep 60;; *) sleep 5;; esac
done; done
while pgrep -f "^python3 -W ignore trellis2" > /dev/null; do sleep 60; done; echo RUN5 DONE
