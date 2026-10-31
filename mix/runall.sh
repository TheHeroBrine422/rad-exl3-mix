#!/bin/bash
# runall.sh -- the full proxy table: all 16 final containers, 8 a card, once every Hessian is on disk.
# Out: $RXM_WORK/hproxy-c0.jsonl, $RXM_WORK/hproxy-c1.jsonl (one line per weight x container); logs $RXM_WORK/hproxy-c*.log
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final} ${RXM_CKPT:=/data/workspace/models/qwen38-27b-bf16}; export RXM_WORK RXM_MODELS RXM_CKPT   # paths (README: Paths)
cd ${RXM_WORK}
until [ $(ls challenge/hessians/calib/*.safetensors 2>/dev/null | wc -l) -ge 257 ] && ! pgrep -f "[h]f download StillDeadcode" > /dev/null; do sleep 30; done
F=${RXM_MODELS}; CK=${RXM_CKPT}; H=challenge/hessians/calib
A=(); B=()
i=0
for t in sc1.40-e3 sc1.60-e3 sc1.80-e3 sc2.00-e3 sc2.20-e3 sc3.00-e4 sc4.00-e4 sc5.00-e5 sc6.00-e6 \
         plain2.00-e3 plain2.50-e3 plain3.00-e4 plain3.50-e4 plain4.00-e4 plain5.00-e5 plain6.00-e6; do
  if (( i++ % 2 )); then B+=("$t=$F/q38-27b-$t.rad"); else A+=("$t=$F/q38-27b-$t.rad"); fi
done
rm -f hproxy-c0.jsonl hproxy-c1.jsonl
ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 ./hproxy/hproxy $CK $H hproxy-c0.jsonl "${A[@]}" > hproxy-c0.log 2>&1 &
ROCR_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 ./hproxy/hproxy $CK $H hproxy-c1.jsonl "${B[@]}" > hproxy-c1.log 2>&1 &
wait
echo HPDONE >> hproxy-c0.log
