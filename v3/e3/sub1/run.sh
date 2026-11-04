#!/bin/bash
# run.sh PART PAR -- #90 sub-1 sweep (README.md). PART a: GDN layers, b: attention layers. One thread a job.
cd /data/work/e3t; export E3_FACTORS=${E3_FACTORS:-0.5,0.7,0.85,1.0,1.15} E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=/data/work/e3t/sub1.jsonl; KS=0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2
case $1 in
  a) T=""; for l in 2 10 30 50 62; do for w in ssm_qkv ssm_z ssm_out ffn_gate_up ffn_down; do T="$T $l:$w"; done; done ;;
  b) T=""; for l in 3 19 43 63; do for w in attn_qg attn_k attn_v attn_output ffn_gate_up ffn_down; do T="$T $l:$w"; done; done ;;
esac
mkdir -p logs
for t in $T; do l=${t%%:*}; w=${t#*:}
  grep -q "\"layer\": $l, \"w\": \"$w\", .*\"bits\": 2.0" $O 2>/dev/null && continue
  echo "CD=0 python3 -W ignore trellis2.py $l $w cubic0.02_i8 $KS 128 $O > logs/sub1-$l-$w.log 2>&1"
done | xargs -P ${2:-12} -I{} bash -c "nice -n 10 env {}"
echo SUB1 DONE $1
