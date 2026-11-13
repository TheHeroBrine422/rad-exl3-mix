#!/bin/bash
# iqrcmp/run.sh CARD -- e2e decode, V3 2.5 vs IQR-V4 iq2r (closest bytes: 10.63 vs 10.89 GB), same box2 card, same
# engine (w137's after-fix build, home g3), w138's bench.py (ctx 0 gen 256 + ctx 8k gen 32, warm 1 + 3 runs), IQR with
# its drafter off (--num-speculative-tokens 0) for the kernel comparison and on for the user-facing number. Rounds
# interleaved (A B A B). Waits for the iq2r download. Lead, 2026-11-13.
C=$1; Q=/data/q27; W=$Q/iqrcmp; T=$Q/w138/tools; I=$Q/rad10-perf/install; H=$Q/w137/g3/home; PORT=$((8390 + C))
IQ=$Q/w138/hf/iq2r/qwen3.8-27b-iq2r-v4.rad; V3=$Q/w138/packs/v3-k2.5-r16k2.5-a8.rad
mkdir -p $W/res; exec >> $W/run.log 2>&1; say() { echo "$(date -u +%T) $*"; }
while pgrep -f "^/usr/bin/python3 /usr/local/bin/hf download" > /dev/null || [ ! -f $IQ ]; do sleep 20; done
say "iq2r: $(stat -c %s $IQ) bytes"
one() { tag=$1 model=$2; shift 2; [ -f $W/res/$tag.jsonl ] && return
  ( flock 5; flock -s 6; ulimit -n 65536
    ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $model --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 \
      --max-num-seqs 8 --host 127.0.0.1 --port $PORT "$@" > $W/res/serve-$tag.log 2>&1 & SP=$!; i=0
    while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i+3)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      python3 -I $T/bench.py --port $PORT --prompts $Q/w137/prompts --out $W/res/$tag.jsonl.part --tag $tag --ctx 0 --warm 1 --runs 3 --gen 256 --server-log $W/res/serve-$tag.log > $W/res/bench-$tag.txt 2>&1 &&
      python3 -I $T/bench.py --port $PORT --prompts $Q/w137/prompts --out $W/res/$tag.jsonl.part --tag $tag --ctx 8k --warm 1 --runs 3 --gen 32 --server-log $W/res/serve-$tag.log >> $W/res/bench-$tag.txt 2>&1 && mv $W/res/$tag.jsonl.part $W/res/$tag.jsonl
      say "$tag: $(tail -n 2 $W/res/bench-$tag.txt | tr '\n' ' ')"
    else say "$tag: no health $(grep -m1 '^E ' $W/res/serve-$tag.log | cut -c1-200)"; fi
    kill -INT $SP 2>/dev/null; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
  ) 5>$Q/locks/card$C.lock 6>$Q/locks/ram.lock; }
for r in 1 2; do one v3-2.5-r$r $V3; one iq2r-nospec-r$r $IQ --num-speculative-tokens 0; done
one iq2r-spec-r1 $IQ
say DONE
