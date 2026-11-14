#!/bin/bash
# e2b.sh CARD ROUND ROW... -- e2e on box2 card CARD (HIP index) under its lock + ram.lock shared, ROW = TAG=HOME=PACK[=ENV]
# (ENV "K=V,K=V" for the server; HOME "-" = rad10-perf's own home): a radiance server (rad10-perf install; w138's #176
# settings: --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 --max-num-seqs 8, + SARGS), w138's bench.py: ctx 8k
# (gen 32) warm 1 + 3 runs, ctx 0 (gen 256) warm 1 + 3 runs -> res/e2/TAG-rROUND.jsonl. Restartable (finished units
# skipped). Log logs/e2.log. (w150, #177-02; w137's e2.sh with packs by path and per-row env)
C=$1; RD=$2; shift 2
Q=/data/q27; W=$Q/w150; R=$W/res/${E2O:-e2}; T=$Q/w138/tools; L=$Q/locks; PORT=$((8470 + C)); I=$Q/rad10-perf/install
mkdir -p $R; exec >> $W/logs/e2.log 2>&1
say() { echo "$(date -u +%T) [card $C] $*"; }
for row in "$@"; do
  IFS='=' read -r tag H P EV <<< "$row"; [ "$H" = - ] && H=$Q/rad10-perf/home; t=$tag-r$RD
  [ -f $R/$t.jsonl ] && continue
  (
    flock 5; flock -s 6; ulimit -n 65536; say "$t: serve home $H pack $P env '$EV'"
    env ${EV//,/ } ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $P --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT $SARGS > $R/serve-$t.log 2>&1 &
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
