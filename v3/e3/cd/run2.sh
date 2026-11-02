#!/bin/bash
# run2.sh MACHINE -- #77 / #78 rebalance (2026-11-02 ~16:55 CST): ffn_down (17,408-wide, ~10x slower) split per K on
# colo; the jobs not yet started (layer 50 gate_up, attention 19 / 43) on Thor. Results -> cd-<machine>.jsonl.
cd /data/work/e3t; export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p logs; O=/data/work/e3t/cd-$1.jsonl
[ "$1" = colo ] && PAR=${PAR:-14} || PAR=${PAR:-12}
if [ "$1" = colo ]; then
  for l in 10 30 50; do for K in 2 2.5 3 4; do
    echo "CD=1 CD_SWEEPS=3 python3 -W ignore trellis2.py $l ffn_down mul1_i8 $K 256 $O > logs/cd-$l-ffn_down-$K.log 2>&1"
    echo "CD=0 python3 -W ignore trellis2.py $l ffn_down gmul1_i8,cubic0.02_i8,cubic0.05_i8 $K 256 $O > logs/cb-$l-ffn_down-$K.log 2>&1"
  done; done
else
  for t in 50:ffn_gate_up 19:attn_qg 19:attn_output 43:attn_qg 43:attn_output; do l=${t%%:*}; w=${t#*:}
    echo "CD=1 CD_SWEEPS=3 python3 -W ignore trellis2.py $l $w mul1_i8 2,2.5,3,4 256 $O > logs/cd-$l-$w.log 2>&1"
    echo "CD=0 python3 -W ignore trellis2.py $l $w gmul1_i8,cubic0.02_i8,cubic0.05_i8 2,2.5,3,4 256 $O > logs/cb-$l-$w.log 2>&1"
  done
fi | xargs -P $PAR -I{} bash -c "nice -n 10 env {}"
echo "RUN2 $1 DONE"
