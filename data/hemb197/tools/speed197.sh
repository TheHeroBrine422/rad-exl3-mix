#!/bin/bash
# speed197.sh NAME MANIFEST "EXTRA" -- #197 (w185, lead note 12:15 AM CST): decode and prefill (TTFT) of one serve row, batch
# 1, greedy, spec off (--num-speculative-tokens 0), 512 tokens, the w178 protocol shortened: contexts 0 / 4k / 16k, 1 warm-up,
# 3 runs (w177's bench.py, read-only). One radiance serve per row (the vload build with the search's plugin home), card
# lock held for the row (CARD, default 0), ram.lock shared. Restartable: a row with a finished NAME.jsonl is skipped.
set -u
ulimit -n 65536   # the vload build keeps every level shard open (see score197.sh)
N=$1; M=$2; X=${3:-}; C=${CARD:-0}; PORT=${PORT:-8291}
W=/data/q27/w185; BIN=$W/rad-vload/install/bin/radiance; HOMEX=/data/q27/rad10-e157b/home
mkdir -p $W/speed $W/logs
[ -s $W/speed/$N.jsonl ] && { echo "$N skipped"; exit 0; }
exec 9>/data/q27/locks/card$C.lock; flock 9
exec 8>/data/q27/locks/ram.lock; flock -s 8
echo "$(date -u +%FT%TZ) speed $N start card $C extra [$X]" >> $W/logs/score.status
ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$HOMEX $BIN --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 \
  --max-num-seqs 8 --host localhost --port $PORT --num-speculative-tokens 0 $X > $W/logs/serve-$N.log 2>&1 &
SP=$!
i=0; while [ $i -lt 900 ] && ! curl -sf -m 2 http://localhost:$PORT/health > /dev/null; do sleep 2; i=$((i + 1)); kill -0 $SP 2>/dev/null || break; done
if curl -sf -m 2 http://localhost:$PORT/health > /dev/null; then
  /data/q27/venv/bin/python -I /data/q27/w177/tools/bench.py --port $PORT --prompts /data/q27/w177/prompts \
    --out $W/speed/$N.jsonl --tag $N --ctx 0,4k,16k --warm 1 --runs 3 --gen 512 --server-log $W/logs/serve-$N.log > $W/logs/bench-$N.log 2>&1
  rc=$?
else
  rc=9; echo "serve did not come up" > $W/logs/bench-$N.log
fi
kill $SP 2>/dev/null; wait $SP 2>/dev/null
echo "$(date -u +%FT%TZ) speed $N rc $rc card $C" >> $W/logs/score.status
exit $rc
