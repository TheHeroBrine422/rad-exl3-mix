#!/bin/bash
# rp.sh CARD ROW... -- #175 (w137) device timestamps of every dispatch of unprofiled 8k prefills: the radiance server as
# pf.sh's (w138's settings, spec off) with RADIANCE_AQL_STAMP=1 (the AQL backend's own completion-signal times, no added
# syncs; rocprofv3's queue interception breaks the AQL backend: 'upload ... device error'), 1 warm + 3 prefill-only 8k
# requests (pq.py), stopped with SIGINT. -> res/stamps-ROW.csv, res/bench-rp-ROW.txt. ROW: v3base (b10b home) | v3on
# (b10v home, FUSE on) | iqrs. kt.py reads it.
C=$1; shift; Q=/data/q27; W=$Q/w137; R=$W/res; L=$Q/locks; PORT=$((8390 + C))
for row in "$@"; do
  [ -f $R/rp-$row.done ] && continue
  H=$W/b10b/home M=$Q/w138/packs/v3-k4.125-r16k4-a8.rad X=() F=0
  [ $row = v3on ] && H=$W/b10v/home F=1
  [ $row = iqrs ] && H=$Q/rad10-perf/home M=$Q/w138/hf/iqr-v4/qwen3.8-27b-iq4r-s-v4.rad X=(--num-speculative-tokens 0)
  (
    flock 5; flock -s 6; rm -f $R/stamps-$row.csv
    EXL3I_A8_FUSE=$F ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H RADIANCE_AQL_STAMP=1 RADIANCE_AQL_STAMP_OUT=$R/stamps-$row.csv \
      $Q/rad10-perf/install/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 \
      --max-num-seqs 8 --host 127.0.0.1 --port $PORT "${X[@]}" > $R/serve-rp-$row.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 1500 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
    python3 -I $W/pq.py $PORT $W/prompts/ctx-8k.txt 4 > $R/bench-rp-$row.txt 2>&1
    kill -INT $SP
    for j in $(seq 300); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
    [ -s $R/stamps-$row.csv ] && touch $R/rp-$row.done
  ) 5>$L/card$C.lock 6>$L/ram.lock
done
