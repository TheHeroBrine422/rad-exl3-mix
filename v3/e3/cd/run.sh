#!/bin/bash
# run.sh -- #77 / #78 sweep on cpu2: one job per (tensor, job type), PAR at once, 2 threads each.
cd /data/work/e3t; export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
O=/data/work/e3t/cd.jsonl; PAR=${PAR:-20}
T="10:ssm_qkv 10:ssm_out 10:ffn_gate_up 10:ffn_down 30:ssm_qkv 30:ssm_out 30:ffn_gate_up 30:ffn_down 50:ssm_qkv 50:ssm_out 50:ffn_gate_up 50:ffn_down 19:attn_qg 19:attn_output 43:attn_qg 43:attn_output"
for t in $T; do l=${t%%:*}; w=${t#*:}
  echo "CD=1 CD_SWEEPS=3 python3 -W ignore trellis2.py $l $w mul1_i8 2,2.5,3,4 256 $O > logs/cd-$l-$w.log 2>&1"
  echo "CD=0 python3 -W ignore trellis2.py $l $w gmul1_i8,cubic0.02_i8,cubic0.05_i8 2,2.5,3,4 256 $O > logs/cb-$l-$w.log 2>&1"
done | (mkdir -p logs; xargs -P $PAR -I{} bash -c "nice -n 10 env {}")
echo CDRUN DONE
