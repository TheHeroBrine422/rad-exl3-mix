#!/bin/bash
# i8hi.sh PAR -- #94: int8-rounded mul1 codebook cost vs mul1 at K6.5..8 (README "#94"). Same 10 tensors as #90-02.
cd /data/work/e3t; export E3_FACTORS=${E3_FACTORS:-0.5,0.7,0.85,1.0,1.15} E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=${I8HI_OUT:-/data/work/e3t/i8hi.jsonl}; KS=6.5,7,7.25,7.5,7.75,8
T="10:ssm_qkv 10:ssm_out 10:ffn_gate_up 10:ffn_down 30:ffn_down 50:ssm_qkv 50:ffn_gate_up 19:attn_qg 19:attn_output 43:attn_v"
mkdir -p logs
for t in $T; do l=${t%%:*}; w=${t#*:}
  grep -q "\"layer\": $l, \"w\": \"$w\", .*mul1_i8.*\"bits\": 8" $O 2>/dev/null && continue
  echo "CD=0 python3 -W ignore trellis2.py $l $w mul1,mul1_i8 $KS 128 $O > logs/i8hi-$l-$w.log 2>&1"
done | xargs -P ${1:-10} -I{} bash -c "nice -n 10 env {}"
echo I8HI DONE
