#!/bin/bash
# run147c.sh -- (w147, #178-02, box2) the decode step's op table, after run147b.sh (one card at a time): per row (B, T4) a
# server with --profile-ops (run147.sh's settings), bench.py ctx 0, warm 0 + 1 run of 256 tokens: the table at exit is
# ~256 decode steps + one ~100-token prefill. Out W/perf/serve-pdec-<row>.log. First free card, held; ram.lock shared.
set -u
Q=/data/q27 W=/data/q27/w147 LK=/data/q27/locks P=$W/perf T=$Q/w138/tools PR=$Q/w137/prompts
say() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/run147.log; }
while ! grep -q "run147b finished" $W/run147.log; do sleep 30; done
while :; do
  C=
  for cc in 0 1 2 3; do exec 9>$LK/card$cc.lock; if flock -n 9; then C=$cc; break; fi; exec 9>&-; done
  [ -n "$C" ] && break; sleep 5
done
exec 6>$LK/ram.lock; flock -s 6
say "run147c: card $C held (pid $$)"
PORT=$((8470 + C))
for row in B T4; do
  t=pdec-$row; [ -f $P/$t.done ] && continue
  ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$W/home $Q/rad10-perf/install/bin/radiance --model $W/mix/2.5-$row.rad --tp 1 \
    --kv-cache-dtype bf16 --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT --profile-ops > $P/serve-$t.log 2>&1 &
  SP=$!; i=0
  while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
  python3 -I $T/bench.py --port $PORT --prompts $PR --out $P/$t.jsonl --tag $t --ctx 0 --warm 0 --runs 1 --gen 256 \
    --server-log $P/serve-$t.log > $P/bench-$t.txt 2>&1
  kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
  grep -q 'per-op timing' $P/serve-$t.log && touch $P/$t.done; say "$t: $(tail -n 1 $P/bench-$t.txt)"
done
exec 9>&-
say "run147c finished"
