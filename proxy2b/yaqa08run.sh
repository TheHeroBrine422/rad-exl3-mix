#!/bin/bash
# yaqa08run.sh -- the 0.8B YAQA dose-response (YAQA08.md), unattended after calib1.sh: search golden, collection with
# snapshots 1K / 4K / 16K, quants (one-sided, two-sided per snapshot) at K2 and K3, evals, paired comparisons.
# Out /data/y08/, summary /data/y08/result.txt.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/y08 G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
YA="--len 1024 --chunk 128 --snap 1024,4096,16384,32768"   # 2048 tokens ran out of memory (gate, 2026-11-02)
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
while pgrep -f "calib1.sh|hcollect.py|setup08.sh" > /dev/null; do sleep 30; done
if [ ! -f $G/kld.json ]; then
  D=$(POOLS=/workspace/rad-exl3-mix/eval/pools/pools.json python3 /workspace/rad-exl3-mix/mix/calib_slice.py /data/challenge/corpus/calib.jsonl 300000 2048 1 search 2>/dev/null)
  python3 $P/mkgold.py $M /data/challenge/calib-meta $G $D > $O/gold.log 2>&1; log "golden g08-search1 done"
fi
if [ ! -f $O/hout/rows.json ]; then
  python3 $P/yaqa08.py $M /data/xcorp/v1/meta $O 32768 $YA --gate > $O/gate.log 2>&1 && grep -q DONE $O/gate.log \
    || { log "GATE FAIL: $(grep -E 'Error|seq' $O/gate.log | tail -2 | tr '\n' ' ')"; exit 1; }
  log "gate: $(grep '^seq' $O/gate.log | tail -1)"
  python3 $P/yaqa08.py $M /data/xcorp/v1/meta $O 32768 $YA > $O/collect.log 2>&1
fi
log "collect: $(grep -E 'DONE|Error' $O/collect.log | tail -1)"
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1)"; }
for k in 2 3; do
  q k${k}one $k
  for n in 1024 4096 16384 32768; do q k${k}two$n $k --hout $O/hout-snap$n; done
  for n in 1024 4096 16384 32768; do log "pair k${k}one -> k${k}two$n: $(python3 $P/pcmp.py $G $O/k${k}one.f32 $O/k${k}two$n.f32)"; done
done
log YAQA08 DONE
