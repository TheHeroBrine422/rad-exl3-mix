#!/bin/bash
# i4run.sh PAR -- #90-02: int4-grid codebook cost vs int8-grid across K0.125..3.5 (where the i4 ladder can replace the
# i8 one). 10 tensors, mul1 rounded to int4 (int4grid) vs to int8 (mul1_i8), same rows / seed / factors.
cd /data/work/e3t; export E3_FACTORS=${E3_FACTORS:-0.5,0.7,0.85,1.0,1.15} E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=/data/work/e3t/i4.jsonl; KS=0.125,0.25,0.5,0.75,1,1.5,2,2.25,2.5,2.75,3,3.5
T="10:ssm_qkv 10:ssm_out 10:ffn_gate_up 10:ffn_down 30:ffn_down 50:ssm_qkv 50:ffn_gate_up 19:attn_qg 19:attn_output 43:attn_v"
mkdir -p logs
for t in $T; do l=${t%%:*}; w=${t#*:}
  grep -q "\"layer\": $l, \"w\": \"$w\", .*int4grid.*\"bits\": 3.5" $O 2>/dev/null && continue
  echo "CD=0 python3 -W ignore trellis2.py $l $w int4grid,mul1_i8 $KS 128 $O > logs/i4-$l-$w.log 2>&1"
done | xargs -P ${1:-10} -I{} bash -c "nice -n 10 env {}"
echo I4 DONE
