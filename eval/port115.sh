#!/bin/bash
# port115.sh [CARD] -- the radiance 1.15.1 port's load test: V2 3.00 scored on golden-ver1 by stock 1.15.1 (--kld-ref,
# our four plugins built from fork branch qc-exl3-115) and by the frozen 1.13 recorder, same container, same day.
# Pass: means within 0.5% (PR #362, in 1.15, moves KLD ~-0.1%). Out /data/port115/
: ${RXM_WORK:=/root}
C=${1:-1} O=/data/port115 M=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad G=$RXM_WORK/golden-ver1
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/result.txt; }
H=$O/home115
if [ ! -d $H ]; then
  cp -a /root/install-115/share/radiance $H
  cp /root/rh115/build-exl3dense/radiance_home/kernels/exl3dense.so /root/rh115/build-exl3r4d/radiance_home/kernels/exl3_r4d.so $H/kernels/
  cp /root/rh115/build-exl3quant/radiance_home/quantizers/exl3quant.so $H/quantizers/
  cp /root/rh115/build-qwen35_exl3/radiance_home/architectures/qwen35_exl3.so $H/architectures/
fi
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
export ROCR_VISIBLE_DEVICES=$C
s=$(date +%s)
RADIANCE_HOME=$H /root/install-115/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
  --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $O/r115.json > $O/r115.log 2>&1
log "1.15.1 rc=$? $(( $(date +%s) - s ))s: $(python3 -c "import json; d=json.load(open('$O/r115.json')); print({k: d[k] for k in list(d)[:6]})" 2>&1 | cut -c1-300)"
s=$(date +%s)
w=$(CARD=$C BIN=$RXM_WORK/radiance-kldgold-fast $RXM_WORK/gold2.sh $M $G $RXM_WORK/corpus/ver1.jsonl $O/r113 bf16 16384)
log "1.13 $w: $(python3 -c "import json; print(json.load(open('$O/r113.json'))['kld'])" 2>&1 | cut -c1-200)"
