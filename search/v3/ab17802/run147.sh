#!/bin/bash
# run147.sh -- (w147, #178-02, box2) the before / after rows on ONE card: the first free one by polling the card locks, held
# on fd 9 to the end, ram.lock shared. rad10-perf's radiance + W/home (this branch's exl3i / exl3dense / exl3quant /
# qwen35_exl3: B's plain a / b path is the code as before, T4 the fused one).
#  1. KLD of W/mix/2.5-{B,T4}.rad on golden-s171d (box2 tools/score.sh: tp 1, bf16 KV, max-model-len 16384, one sequence)
#  2. time, w144's perf144.sh protocol (w137's #175 / pf.sh): a server (--tp 1 --kv-cache-dtype bf16 --max-model-len 40960
#     --max-num-seqs 8), per row  e2e: pq.py 8k prefill-only x 4, bench.py ctx 8k (gen 32) and ctx 0 (gen 256), warm 1 +
#     3 runs;  prof: --profile-ops, pq.py 8k x 2 (the per-op table at exit). Two rounds, r1 B then T4, r2 T4 then B.
# Out W/scores/, W/perf/. Restartable: finished scores / (row, mode, round) skipped.
set -u
Q=/data/q27 W=/data/q27/w147 LK=/data/q27/locks P=$W/perf T=$Q/w138/tools PQ=$Q/w137/pq.py PR=$Q/w137/prompts
G=$W/gold/golden-s171d; mkdir -p $P $W/scores
say() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/run147.log; }
while :; do
  C=
  for cc in 0 1 2 3; do exec 9>$LK/card$cc.lock; if flock -n 9; then C=$cc; break; fi; exec 9>&-; done
  [ -n "$C" ] && break; sleep 5
done
exec 6>$LK/ram.lock; flock -s 6
say "card $C held (pid $$)"
for n in B T4; do
  o=$W/scores/2.5-$n-s171d; [ -f $o.json.rows ] && continue
  bash $Q/tools/score.sh $Q/rad10-perf/install $W/home $W/mix/2.5-$n.rad $G $o $C | tee -a $W/run147.log
  say "score $n: $(grep -E 'static weights|I kld: [0-9]+ / ' $o.log | tr '\n' '|')"
done
PORT=$((8470 + C))
for R in 1 2; do
  rows="B T4"; [ $R = 2 ] && rows="T4 B"
  for row in $rows; do
    M=$W/mix/2.5-$row.rad
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
done
exec 9>&-
say "run147 finished"
