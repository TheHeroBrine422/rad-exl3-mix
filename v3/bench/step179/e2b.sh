#!/bin/bash
# e2b.sh CARD ROUND ROW... -- w150's e2b.sh (#177-02) with this worker's dirs: ROW = TAG=HOME=PACK[=ENV] (ENV "K=V,K=V"),
# a radiance server (rad10-perf install; --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 --max-num-seqs 8), w138's
# bench.py ctx 8k (gen 32) + ctx 0 (gen 256), warm 1 + 3 runs each -> res/e2r/TAG-rROUND.jsonl; under card + ram (shared)
# locks; finished units skipped. Log logs/e2r.log. (w152: the #177-02 decode re-measure after fdd480c)
C=$1; RD=$2; shift 2
Q=/data/q27; W=$Q/w152; R=$W/res/e2r; T=$Q/w138/tools; L=$Q/locks; PORT=$((8470 + C)); I=$Q/rad10-perf/install
mkdir -p $R; exec >> $W/logs/e2r.log 2>&1
say() { echo "$(date -u +%T) [card $C] $*"; }
for row in "$@"; do
  IFS='=' read -r tag H P EV <<< "$row"; t=$tag-r$RD
  [ -f $R/$t.jsonl ] && continue
  (
    flock 5; flock -s 6; ulimit -n 65536; say "$t: serve home $H pack $P env '$EV'"
    env ${EV//,/ } ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $P --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT > $R/serve-$t.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      meta="{\"row\":\"$tag\",\"home\":\"$H\",\"pack\":\"$P\",\"env\":\"$EV\",\"card_hip\":$C}"
      python3 -I $T/bench.py --port $PORT --prompts $Q/w137/prompts --out $R/$t.jsonl.part --tag $t --ctx 8k --warm 1 --runs 3 --gen 32 \
        --server-log $R/serve-$t.log --meta "$meta" > $R/bench-$t.txt 2>&1 &&
      python3 -I $T/bench.py --port $PORT --prompts $Q/w137/prompts --out $R/$t.jsonl.part --tag $t --ctx 0 --warm 1 --runs 3 --gen 256 \
        --server-log $R/serve-$t.log --meta "$meta" >> $R/bench-$t.txt 2>&1 && mv $R/$t.jsonl.part $R/$t.jsonl
      kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
      say "$t: done ($(tail -n 1 $R/bench-$t.txt))"
    else kill $SP 2>/dev/null; wait $SP 2>/dev/null; say "$t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-200))"; fi
  ) 5>$L/card$C.lock 6>$L/ram.lock
done
