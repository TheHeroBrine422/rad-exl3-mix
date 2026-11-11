#!/bin/bash
# s129b.sh -- #129-02: score mix-k6{c9,swj} on golden-search27b (x + c), then docstat over 27b and 27b+27c pooled.
set -u
A=/data/w99 I8=/data/install-116 H8=/data/swap27/home RXM_WORK=/root P=/data/venv-exl3/bin/python
pick() { for d in /data/w91/golden-search27b-$1 /data/w90/golden-search27b-$1; do [ -f $d/tokens.i32 ] && [ -f $d/rows.f32 ] && { echo $d; return; }; done; }
GX=$(pick x) GC=$(pick c); echo "goldens $GX $GC" | tee -a $A/s129b.txt
for tag in k6c9 k6swj; do for part in x c; do
  G=$GX; [ $part = c ] && G=$GC; o=$A/scores/$tag-27b-$part
  [ -f $o.json.rows ] && continue
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    s=$(date +%s); ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H8 $I8/bin/radiance --model $A/mix/mix-$tag.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $A/logs/score-$tag-27b-$part.log 2>&1; rc=$?
    [ $rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo "$(date -u +%H:%M:%S) score $tag-27b-$part rc=$rc $(( $(date +%s) - s ))s" | tee -a $A/s129b.txt )
  [ -f $o.json.rows ] || { echo "S129B FAIL $tag $part" | tee -a $A/s129b.txt; exit 1; }
done; done
S=$A/scores
echo "== 27b only" >> $A/s129b.txt
$P /data/docstat/docstat.py $GX $GC $S/k6c9-27b-x.json.rows $S/k6c9-27b-c.json.rows $S/k6swj-27b-x.json.rows $S/k6swj-27b-c.json.rows >> $A/s129b.txt 2>&1
echo "S129B DONE" >> $A/s129b.txt
