#!/bin/bash
# tsweep1.sh -- trellis check (E3b): 4 codebooks x K2..6 on 128 sampled rows of 4 tensors; starts when sweep1 ends
: ${RXM_LOGS:=/workspace/logs}; export RXM_LOGS   # paths (README: Paths)
cd "$(dirname "$0")"
until grep -q SWEEP1DONE ${RXM_LOGS}/e3-sweep1.err; do sleep 60; done
for lt in "10 ssm_z" "43 attn_output" "10 ssm_out" "43 ffn_gate_up"; do
  OMP_NUM_THREADS=16 python3 -W ignore trellis.py $lt mul1,mul1_i8,gauss_i8,int4grid 2,3,4,5,6 128 tsweep1.jsonl > /dev/null 2>> ${RXM_LOGS}/e3-tsweep1.err
done
echo TSWEEP1DONE >> ${RXM_LOGS}/e3-tsweep1.err
