#!/bin/bash
# a1run.sh "LAYERS" P_UQ P_DOWN -- #81 driver (one machine): waits for running trellis2 jobs (#79 / #80), then stage 1
# (gate_up rows in 1024-row jobs, K 2 / 2.5 / 3, P_UQ at once, 1 thread each), then stage 2 (down arms D, ug, u,
# ugtrue per layer and K, P_DOWN at once: ~15 GB each). Layers in the given order (L30 first: the pass-rule layer).
# Restart-safe (finished parts / result lines are skipped). Out: /data/work/a1/{gu,a1.jsonl}, logs /data/work/a1/logs.
cd /data/work/e3t; export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data/work/a1; mkdir -p $A/gu $A/logs; L="$1"; PU=${2:-30}; PD=${3:-8}
while pgrep -f "^python3 -W ignore trellis2" > /dev/null; do sleep 60; done
echo "$(date -u +%T) stage 1"
for l in $L; do for K in 2 2.5 3; do for r in $(seq 0 1024 34815); do
  [ -f $A/gu/L$l-K$K/gu-$(printf %05d $r).npy ] || echo "E3_FACTORS=1.0 python3 -W ignore a1test.py uq $l $K $r $((r + 1024)) $A/gu >> $A/logs/uq-$l-$K.log 2>&1"
done; done; done | xargs -P $PU -I{} nice -n 10 bash -c "{}"
echo "$(date -u +%T) stage 2"
have() { [ -f $A/a1.jsonl ] && grep -q "\"layer\": $1, \"w\": \"ffn_down\", \"arm\": \"$3\", \"K\": $(python3 -c "print(float('$2'))")," $A/a1.jsonl; }
for l in $L; do for K in 2 2.5 3; do for arm in D ug u ugtrue; do
  have $l $K $arm || echo "E3_FACTORS=1.0,1.15 python3 -W ignore a1test.py down $l $K $arm $A/gu $A/a1.jsonl >> $A/logs/down-$l-$K-$arm.log 2>&1"
done; done; done | xargs -P $PD -I{} nice -n 10 bash -c "{}"
echo "$(date -u +%T) A1RUN DONE"
