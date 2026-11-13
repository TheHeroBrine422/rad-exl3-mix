#!/bin/bash
# perf144.sh ROUND ROW... -- (w144, #174-02, box2) VRAM as served, 8k prefill and decode step of W/mix/<ROW>.rad (k4-l3,
# k4-t2.5, k4-t4, ... ; bf16 = the trunk as built, w138's v3-k4.125-r16k4-a8.rad: bf16 table host-pinned), w137's #175
# protocol (pf.sh): a radiance server (rad10-perf install + W/home, --tp 1 --kv-cache-dtype bf16 --max-model-len 40960
# --max-num-seqs 8), per row
#   e2e : pq.py 8k prefill-only x 4 (w137), then w138's bench.py ctx 8k (gen 32) and ctx 0 (gen 256), warm 1 + 3 runs
#   prof: --profile-ops, pq.py 8k x 2 (the per-op table at exit: embed_exl3i / embed_tq / embed_lookup us a call)
# All rows of a call on one card (the first free one by polling the card locks, held throughout), ram.lock shared. Out
# W/perf/<mode>-<row>-r<ROUND>.*; the load's VRAM report is in the serve log. Restartable: finished (row, mode) skipped.
set -u
Q=/data/q27 W=/data/q27/w144 LK=/data/q27/locks P=$W/perf T=$Q/w138/tools PQ=$Q/w137/pq.py PR=$Q/w137/prompts
R=$1; shift; mkdir -p $P $W/run
say() { echo "$(date -u +%T) $*" | tee -a $P/perf144.log; }
while :; do                          # the first free card by polling its lock, held on fd 9 until the end
  C=
  for cc in 0 1 2 3; do exec 9>$LK/card$cc.lock; if flock -n 9; then C=$cc; break; fi; exec 9>&-; done
  [ -n "$C" ] && break; sleep 2
done
PORT=$((8390 + C))
exec 6>$LK/ram.lock; flock -s 6
for row in "$@"; do
  M=$W/mix/$row.rad; [ $row = bf16 ] && M=$Q/w138/packs/v3-k4.125-r16k4-a8.rad
  for mode in e2e prof; do
    t=$mode-$row-r$R; [ -f $P/$t.done ] && continue
    X=(); [ $mode = prof ] && X=(--profile-ops)
    say "$t: serve $(basename $M) card $C"
    ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$W/home $Q/rad10-perf/install/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT "${X[@]}" > $P/serve-$t.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      if [ $mode = prof ]; then
        python3 -I $PQ $PORT $PR/ctx-8k.txt 2 > $P/bench-$t.txt 2>&1
      else
        python3 -I $PQ $PORT $PR/ctx-8k.txt 4 > $P/bench-$t.txt 2>&1
        python3 -I $T/bench.py --port $PORT --prompts $PR --out $P/$t.jsonl.part --tag $t --ctx 8k --warm 1 --runs 3 --gen 32 \
          --server-log $P/serve-$t.log >> $P/bench-$t.txt 2>&1 &&
        python3 -I $T/bench.py --port $PORT --prompts $PR --out $P/$t.jsonl.part --tag $t --ctx 0 --warm 1 --runs 3 --gen 256 \
          --server-log $P/serve-$t.log >> $P/bench-$t.txt 2>&1 && mv $P/$t.jsonl.part $P/$t.jsonl
      fi
      kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
      if [ $mode = prof ]; then grep -q 'per-op timing' $P/serve-$t.log && touch $P/$t.done; else [ -f $P/$t.jsonl ] && touch $P/$t.done; fi
      say "$t: done | $(grep -E 'static weights' $P/serve-$t.log | head -n 1 | sed 's/ \+/ /g') | $(tail -n 1 $P/bench-$t.txt)"
    else kill $SP 2>/dev/null; wait $SP 2>/dev/null; say "$t: no health ($(grep -m1 '^E ' $P/serve-$t.log | cut -c1-200))"; fi
  done
done
exec 9>&-
say "perf r$R $* finished"
