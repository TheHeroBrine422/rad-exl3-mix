#!/bin/bash
# e2.sh CARD ROUND PACK|/PACK/PATH.rad... -- ROWS env "name=HOME ...": e2e on box2 card CARD (HIP index) under its lock + ram.lock shared,
# per pack and row one radiance server (rad10-perf install, --tp 1 --kv-cache-dtype bf16 --max-model-len 40960
# --max-num-seqs 8, as w137's e2.sh / #176), w138's bench.py: ctx 8k (gen 64) warm 1 + 3 runs, ctx 0 (gen 256) warm 1 +
# 3 runs -> res/e2/PACK-ROW-rROUND.jsonl. Env SRVENV: VAR=VALUE words for the server (e.g. QWEN35_EXL3_FUSE=0).
# Restartable (a finished unit is skipped). Log logs/e2.log. (w152, #179)
C=$1; RD=$2; shift 2
Q=/data/q27; W=$Q/w152; R=$W/res/e2; T=$Q/w138/tools; L=$Q/locks; PORT=$((8470 + C)); I=$Q/rad10-perf/install
mkdir -p $R; exec >> $W/logs/e2.log 2>&1; ulimit -n 65536
say() { echo "$(date -u +%T) [card $C] $*"; }
for P in "$@"; do for rh in $ROWS; do
  row=${rh%%=*}; H=${rh#*=}; M=$Q/w138/packs/$P.rad; PN=$P
  case $P in /*) M=$P; PN=$(basename $P .rad) ;; esac   # a pack by path
  t=$PN-$row-r$RD
  [ -f $R/$t.jsonl ] && continue
  (
    flock 5; flock -s 6; say "$t: serve home $H ${SRVENV:-}"
    env $SRVENV ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT > $R/serve-$t.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      meta="{\"row\":\"$row\",\"home\":\"$H\",\"pack\":\"$PN\",\"card_hip\":$C}"
      python3 -I $T/bench.py --port $PORT --prompts $Q/w137/prompts --out $R/$t.jsonl.part --tag $t --ctx 8k --warm 1 --runs 3 --gen 64 \
        --server-log $R/serve-$t.log --meta "$meta" > $R/bench-$t.txt 2>&1 &&
      python3 -I $T/bench.py --port $PORT --prompts $Q/w137/prompts --out $R/$t.jsonl.part --tag $t --ctx 0 --warm 1 --runs 3 --gen 256 \
        --server-log $R/serve-$t.log --meta "$meta" >> $R/bench-$t.txt 2>&1 && mv $R/$t.jsonl.part $R/$t.jsonl
      kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
      say "$t: done ($(tail -n 1 $R/bench-$t.txt))"
    else kill $SP 2>/dev/null; wait $SP 2>/dev/null; say "$t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-200))"; fi
  ) 5>$L/card$C.lock 6>$L/ram.lock
done; done
say "e2 round $RD DONE"
