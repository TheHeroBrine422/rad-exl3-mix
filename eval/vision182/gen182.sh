#!/bin/bash
# gen182.sh -- (ds8, card CARD=7 under its lock + ram.lock shared) the #182 answers: IQR-V4 S (a third-party quant, neutral
# to every candidate) served by rad10-perf + w153's home-r with the tower (--mm-max-patches 4096), gen182.py over the image
# items and the text control. Restartable (gen182.py skips finished ids). Log W/logs/gen182.log. (w156)
W=/data/q27/w156; Q=/data/q27; I=$Q/rad10-perf/install; H=$Q/w153/home-r; M=$Q/w153/ref/qwen3.8-27b-iq4r-s-v4.rad; CARD=${CARD:-7}
PORT=$((8180 + CARD)); L=$Q/locks; mkdir -p $W/gen $W/logs
exec >> $W/logs/gen182.log 2>&1
say() { echo "$(date -u +%T) $*"; }
( flock 5; flock -s 4; say "card $CARD held"
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
    --max-num-seqs 8 --mm-max-patches 4096 --num-speculative-tokens 0 --host 127.0.0.1 --port $PORT > $W/gen/serve.log 2>&1 &
  SP=$!; i=0
  while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
  if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
    say "server up"
    python3 -I $W/tools/gen182.py $PORT $W/set/items.jsonl $W/set/img $W/gen/ans-img.jsonl 512 8 > $W/gen/gen-img.txt 2>&1
    python3 -I $W/tools/gen182.py $PORT $W/set/text-items.jsonl $W/set/img $W/gen/ans-text.jsonl 512 8 > $W/gen/gen-text.txt 2>&1
    say "gen: $(wc -l < $W/gen/ans-img.jsonl) image, $(wc -l < $W/gen/ans-text.jsonl) text answers"
  else say "no health: $(grep -m1 '^E ' $W/gen/serve.log | cut -c1-200)"; fi
  kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
  say "card $CARD released" ) 5>$L/card$CARD.lock 4>$L/ram.lock
