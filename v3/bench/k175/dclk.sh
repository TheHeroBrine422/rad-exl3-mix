#!/bin/bash
# dclk.sh CARD ROW... -- #175 (w137) decode at batch 1 with clocks: a radiance server as pf.sh's (w138's #176 settings, spec
# off), bench.py ctx 0, 1 warm + 2 runs of 512 tokens, rocm-smi sampled meanwhile -> res/dclk-ROW.jsonl, res/smi-dclk-ROW.txt.
# ROW: v3base (b10b home) | iqrs. Card lock + ram.lock shared. Skips finished rows.
C=$1; shift; Q=/data/q27; W=$Q/w137; R=$W/res; T=$Q/w138/tools; L=$Q/locks; PORT=$((8380 + C)); SI=$((3 - C))
for row in "$@"; do
  [ -f $R/dclk-$row.jsonl ] && continue
  H=$W/b10b/home M=$Q/w138/packs/v3-k4.125-r16k4-a8.rad X=()
  [ $row = iqrs ] && H=$Q/rad10-perf/home M=$Q/w138/hf/iqr-v4/qwen3.8-27b-iq4r-s-v4.rad X=(--num-speculative-tokens 0)
  (
    flock 5; flock -s 6
    ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $Q/rad10-perf/install/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT "${X[@]}" > $R/serve-dclk-$row.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
    ( while :; do echo "T $(date +%s.%N)"; rocm-smi -d $SI --showclocks --showpower --showtemp 2>/dev/null | grep -E "sclk|mclk|Power \\(|edge|junction"; done ) > $R/smi-dclk-$row.txt &
    SM=$!
    python3 -I $T/bench.py --port $PORT --prompts $W/prompts --out $R/dclk-$row.jsonl.part --tag dclk-$row --ctx 0 --warm 1 --runs 2 \
      --gen 512 --server-log $R/serve-dclk-$row.log > $R/bench-dclk-$row.txt 2>&1 && mv $R/dclk-$row.jsonl.part $R/dclk-$row.jsonl
    kill $SM; wait $SM 2>/dev/null
    kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
  ) 5>$L/card$C.lock 6>$L/ram.lock
done
