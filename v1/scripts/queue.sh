#!/bin/bash
# queue.sh -- THE ONLY thing that runs the recorder (gold.sh kills any other): every GPU job, one at a time.
# One JSON line a job to /root/results.jsonl: set (100k | long), model, emb (bf16 = the checkpoint's table), metrics.
cd /root
R=/root/results.jsonl
while pgrep -f "^/bin/bash /root/runlong.sh" >/dev/null; do sleep 20; done   # the long-context ladder in flight

rec() {  # SET MODEL EMB CONTAINER
  local set=$1 model=$2 emb=$3 m=$4 o=/root/q-$1-$2-$3 w
  if [ "$set" = 100k ]; then
    w=$(RADIANCE_KLD_MAXTOK=4096 ./gold.sh $m golden100k corpus100k.jsonl $o bf16 8192 --max-num-batched-tokens 4096)
  else
    w=$(./gold.sh $m goldenlong corpuslong.jsonl $o bf16 131072 --max-num-batched-tokens 4096)
  fi
  python3 -c "
import json; r=json.load(open('$o.json'))
d={'set':'$set','model':'$model','emb':'$emb','run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'median':r['kld']['median'],
   'p99':r['kld']['p99'],'p99_9':r['kld']['p99_9'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio']}
if '$set'=='long': d['long_by_pos']={k:round(v['kld'],5) for k,v in r['by_position_long_docs'].items()}
print(json.dumps(d))" >> $R 2>/dev/null || echo "{\"set\":\"$set\",\"model\":\"$model\",\"emb\":\"$emb\",\"run\":\"$w\",\"fail\":1}" >> $R
  rm -rf $o
}
sweep() {  # TAG FMT... -- wait for each prepared copy, score it, delete it
  local t=$1; shift
  for f in "$@"; do
    local m=/root/emb/$t-$f.rad
    while [ ! -f $m ] && ! grep -q "^$t $f FAIL" /root/emb/prep.log; do sleep 20; done
    [ -f $m ] && { rec 100k $t $f $m; rm -f $m; } || echo "{\"set\":\"100k\",\"model\":\"$t\",\"emb\":\"$f\",\"fail\":\"prep\"}" >> $R
  done
}
FMTS="fp8t i8g128 i4g128s i4g128sh i4g32s i3g128sh i2g128sh"

# the self-calibrated 4.00: prepare its variants on the CPU as soon as its container exists (no GPU)
( while ! grep -q "4.00sc rc=" /root/conv.out 2>/dev/null; do sleep 30; done
  [ -f /root/models/q38-27b-exl3-4.00scbpw.rad ] && nice /root/embprep.sh /root/models/q38-27b-exl3-4.00scbpw.rad 4.00sc $FMTS ) &

# 1. the ladder's gaps (killed by the queue collision)
rec long 1.40 bf16 /root/models/q38-27b-exl3-1.40bpw.rad
rec 100k 6.00 bf16 /root/models/q38-27b-exl3-6.00bpw.rad
rec 100k 3.00 bf16 /root/models/q38-27b-exl3-3.00bpw.rad        # crashed once (rc=139): reproducible?
# 2. 2.00: the embedding formats (bf16 baseline: r100k-2.00.json), then the real fp8 gather (embed_lookup_q)
rec 100k 2.00 bf16 /root/models/q38-27b-exl3-2.00bpw.rad        # baseline again, same queue (exactness check)
sweep 2.00 $FMTS
while ! grep -q "fp8real rc=" /root/conv.out 2>/dev/null; do sleep 30; done
[ -f /root/emb/2.00-fp8real.rad ] && rec 100k 2.00 fp8real /root/emb/2.00-fp8real.rad
# 3. the self-calibrated 4.00: baselines, then the formats
while ! grep -q "4.00sc rc=" /root/conv.out 2>/dev/null; do sleep 30; done
if [ -f /root/models/q38-27b-exl3-4.00scbpw.rad ]; then
  rec 100k 4.00sc bf16 /root/models/q38-27b-exl3-4.00scbpw.rad
  rec long 4.00sc bf16 /root/models/q38-27b-exl3-4.00scbpw.rad
  sweep 4.00sc $FMTS
fi
wait
echo QUEUEDONE >> $R
