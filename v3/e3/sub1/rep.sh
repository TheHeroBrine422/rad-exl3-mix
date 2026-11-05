#!/bin/bash
# rep.sh PAR -- #90-03 (README): seed-1 replicate of #90's 49 tensors, then i4 vs i8 on 10 new tensors. One thread a job.
cd /data/work/e3t; export E3_FACTORS=${E3_FACTORS:-0.5,0.7,0.85,1.0,1.15} E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
{ O=/data/work/e3t/sub1-s1.jsonl; KS=0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2
  for l in 2 10 30 50 62; do for w in ssm_qkv ssm_z ssm_out ffn_gate_up ffn_down; do echo "$l:$w"; done; done
  for l in 3 19 43 63; do for w in attn_qg attn_k attn_v attn_output ffn_gate_up ffn_down; do echo "$l:$w"; done; done; } | while read t; do l=${t%%:*}; w=${t#*:}
  grep -q "\"layer\": $l, \"w\": \"$w\", .*\"bits\": 2.0" /data/work/e3t/sub1-s1.jsonl 2>/dev/null && continue
  echo "E3_SEED=1 E3_ROWSEED=1 CD=0 python3 -W ignore trellis2.py $l $w cubic0.02_i8 0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2 128 /data/work/e3t/sub1-s1.jsonl > logs/sub1s1-$l-$w.log 2>&1"
done | xargs -P ${1:-16} -I{} bash -c "nice -n 10 env {}"
for t in 6:ssm_qkv 6:ffn_gate_up 26:ssm_out 26:ffn_down 46:ssm_z 46:ffn_gate_up 58:ssm_qkv 39:attn_qg 39:attn_output 39:ffn_down; do l=${t%%:*}; w=${t#*:}
  grep -q "\"layer\": $l, \"w\": \"$w\", .*int4grid.*\"bits\": 3.5" /data/work/e3t/i4b.jsonl 2>/dev/null && continue
  echo "CD=0 python3 -W ignore trellis2.py $l $w int4grid,mul1_i8 0.125,0.25,0.5,0.75,1,1.5,2,2.25,2.5,2.75,3,3.5 128 /data/work/e3t/i4b.jsonl > logs/i4b-$l-$w.log 2>&1"
done | xargs -P ${1:-16} -I{} bash -c "nice -n 10 env {}"
echo REP DONE
