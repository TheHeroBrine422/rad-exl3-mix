#!/bin/bash
# pub208.sh (box2) -- #208 public eval (the published numbers; run once, after the picks are frozen and mad-only decided, never used
# to select or tune): each verified release container /data/q27/release-v3/vram/<size>.rad on the whole public golden (596 docs,
# 1.69M rows, incl. the 2 long agentic traces at full length) with --tp 2 --max-model-len 262144 on cards $CARDS (both locks held).
# Output /data/q27/release-v3/public/<size>.json(.rows,.log). Restartable. Stop: touch /data/q27/release-v3/STOP-pub.
set -u
R=/data/q27/release-v3; Q=/data/q27; NEW=$Q/w201/rad21; G=$Q/challenge/golden/public; CARDS=${CARDS:-1,2}
ORDER="2 2.5 2.75 3 3.25 3.5 3.75 4 4.5 5 5.5 6 6.5 7 1.5 1.75"; mkdir -p $R/public
c1=${CARDS%,*}; c2=${CARDS#*,}
for S in $ORDER; do
  while [ ! -f $R/state/$S.ok ]; do [ -f $R/STOP-pub ] && exit 0; [ -f $R/state/$S.bad ] && break; sleep 60; done
  [ -f $R/state/$S.ok ] || continue; o=$R/public/$S; [ -f $o.json ] && continue; [ -f $R/STOP-pub ] && exit 0
  t0=$(date +%s)
  flock $Q/locks/card$c1.lock flock $Q/locks/card$c2.lock flock -s $Q/locks/ram.lock env ROCR_VISIBLE_DEVICES=$CARDS RADIANCE_HOME=$NEW/home \
    $NEW/install/bin/radiance --model $R/vram/$S.rad --tp 2 --kv-cache-dtype bf16 --max-model-len 262144 --max-num-seqs 1 \
    --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
  echo "$(date -u +%H:%M:%S) public $S rc=$? $(( $(date +%s) - t0 )) s $(grep -o '"kld": [0-9.e-]*' $o.json 2>/dev/null | head -1)" >> $R/pub208.log
done
echo "$(date -u +%H:%M:%S) ALL DONE" >> $R/pub208.log
