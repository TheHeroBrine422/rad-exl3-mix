#!/bin/bash
# pf.sh CARD MODE ROW... -- #175 (w137) e2e prefill at 8k + decode at batch 1, and --profile-ops of an 8k prefill, on box2
# card CARD (HIP index) under its lock + ram.lock shared. A radiance server as w138's #176 runs it (--tp 1 --kv-cache-dtype
# bf16 --max-model-len 40960 --max-num-seqs 8, spec off), w138's bench.py as the client (OpenAI chat, greedy, a nonce line
# per request: no prefix-cache hit).
#   MODE e2e[N]  : round N: ctx 8k (gen 32) warm 1 + 3 runs, ctx 0 (gen 256) warm 1 + 3 runs  -> res/e2e-ROW-rN.jsonl
#   MODE profN   : --profile-ops, N prefill-only 8k requests (pq.py)                          -> res/serve-profN-ROW.log
#                  (prof1 and prof5: the per-op difference is 4 warm prefills, profsum8k.py)
#   MODE clk     : 1 warm + 6 prefill-only 8k requests (pq.py) with rocm-smi sampled ~3 / s (sclk, mclk, power, temp)
#                  -> res/bench-clk-ROW.txt, res/smi-clk-ROW.txt (smisum.py)
# ROW: v3today (rad10-perf install + home, 5c9956c) | v3base (the same source built by plugb.sh: b10b, device code = today's) | v3off / v3on (rad10-perf install + b10v home, EXL3I_A8_FUSE 0 / 1)
#      | iqrs (IQR-V4 S on rad10-perf). Restartable: a finished (ROW, MODE) is skipped. Log logs/pf.log.
C=$1; MODE=$2; shift 2
Q=/data/q27; W=$Q/w137; R=$W/res; T=$Q/w138/tools; L=$Q/locks; PORT=$((8370 + C))
V3=${V3:-$Q/w138/packs/v3-k4.125-r16k4-a8.rad}; IQRS=$Q/w138/hf/iqr-v4/qwen3.8-27b-iq4r-s-v4.rad
mkdir -p $R; exec >> $W/logs/pf.log 2>&1
say() { echo "$(date -u +%T) [card $C] $*"; }
for row in "$@"; do
  I=$Q/rad10-perf/install H=$Q/rad10-perf/home F=0 M=$V3 X=()
  case $row in
    v3today) ;;
    v3base) H=$W/b10b/home ;;
    v3off) H=$W/b10v/home ;;
    v3on) H=$W/b10v/home F=1 ;;
    iqrs) M=$IQRS X=(--num-speculative-tokens 0) ;;
    *) say "bad row $row"; continue ;;
  esac
  case $MODE in
    e2e*) t=e2e-$row-r${MODE#e2e}; [ -f $R/$t.jsonl ] && continue ;;
    clk) t=clk-$row; [ -f $R/$t.done ] && continue ;;
    prof*) t=$MODE-$row; X+=(--profile-ops); [ -f $R/$t.done ] && continue ;;
    *) say "bad mode $MODE"; exit 2 ;;
  esac
  (
    flock 5; flock -s 6; say "$t: serve $(basename $M) home $H FUSE $F"
    EXL3I_A8_FUSE=$F ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT "${X[@]}" > $R/serve-$t.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      meta="{\"row\":\"$row\",\"fuse\":$F,\"home\":\"$H\",\"pack\":\"$(basename $M)\",\"card_hip\":$C}"
      if [ $MODE = clk ]; then
        SI=$((3 - C)); python3 -I $W/pq.py $PORT $W/prompts/ctx-8k.txt 1 > $R/bench-$t.txt 2>&1
        ( while :; do echo "T $(date +%s.%N)"; rocm-smi -d $SI --showclocks --showpower --showtemp 2>/dev/null | grep -E "sclk|mclk|Power \(|edge|junction"; done ) > $R/smi-$t.txt &
        SM=$!; python3 -I $W/pq.py $PORT $W/prompts/ctx-8k.txt 6 | while read l; do echo "$(date +%s.%N) $l"; done >> $R/bench-$t.txt 2>&1
        kill $SM; wait $SM 2>/dev/null; touch $R/$t.done
      elif [[ $MODE = prof* ]]; then
        python3 -I $W/pq.py $PORT $W/prompts/ctx-8k.txt ${MODE#prof} > $R/bench-$t.txt 2>&1
      else
        python3 -I $T/bench.py --port $PORT --prompts $W/prompts --out $R/$t.jsonl.part --tag $t --ctx 8k --warm 1 --runs 3 --gen 32 \
          --server-log $R/serve-$t.log --meta "$meta" > $R/bench-$t.txt 2>&1 &&
        python3 -I $T/bench.py --port $PORT --prompts $W/prompts --out $R/$t.jsonl.part --tag $t --ctx 0 --warm 1 --runs 3 --gen 256 \
          --server-log $R/serve-$t.log --meta "$meta" >> $R/bench-$t.txt 2>&1 && mv $R/$t.jsonl.part $R/$t.jsonl
      fi
      kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
      [[ $MODE = prof* ]] && grep -q 'per-op timing' $R/serve-$t.log && touch $R/$t.done
      say "$t: done ($(tail -n 1 $R/bench-$t.txt))"
    else kill $SP 2>/dev/null; wait $SP 2>/dev/null; say "$t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-200))"; fi
  ) 5>$L/card$C.lock 6>$L/ram.lock
done
say "pf $MODE $* finished"
