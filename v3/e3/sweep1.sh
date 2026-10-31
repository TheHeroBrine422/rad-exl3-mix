#!/bin/bash
# sweep1.sh -- E3 sweep: layers 10 (GDN) and 43 (attention), every linear, bits 2..8, the four methods that matter.
# Two workers x 8 threads; results -> sweep1.jsonl next to this script
: ${RXM_LOGS:=/workspace/logs}; export RXM_LOGS   # paths (README: Paths)
cd "$(dirname "$0")"
M=rtn_rot,gptq_rot,gptq_g128i8,lloyd_rot
{ for t in ssm_qkv ssm_z ssm_out ffn_gate_up ffn_down; do echo "10 $t"; done
  for t in attn_qg attn_k attn_v attn_output ffn_gate_up ffn_down; do echo "43 $t"; done; } |
  OMP_NUM_THREADS=8 xargs -P 2 -L 1 sh -c 'python3 -W ignore e3.py $0 $1 '$M' 2,3,4,5,6,8 sweep1.jsonl > /dev/null 2>> '${RXM_LOGS}/e3-sweep1.err
echo SWEEP1DONE >> ${RXM_LOGS}/e3-sweep1.err
