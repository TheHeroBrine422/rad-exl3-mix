#!/bin/bash
# gpu157.sh STAGE... -- (w157, #174-03, box2 card 1) the GPU units of RULES §0-§3 / §5. Every unit takes card 1's lock
# (exclusive) and ram.lock (shared) for its own duration and skips when its output exists. Build e1 = qc-exl3-v3-emb157
# (mk157.sh: b10e1 / b714e1). Log W/logs/gpu157.log. Stages:
#   gtest   for TC 10 and 714: exl3dense_embed_exl3i_gpu_test (device == host == oracle; then --host: the table in mapped
#           pinned host memory) and w152's ctall.sh (every exl3i device suite) -> W/res/gt/
#   kld A.. KLD of W/mix/<A>.rad on golden-s173e (rad10-perf radiance + b10e1/home; tp 1, bf16 KV, 16k, one sequence)
#           -> W/res/kld-<A>.json(.rows,.log)
#   perf    B and T: w144's perf144.sh protocol (pq.py 8k prefill x 4, bench.py ctx 8k gen 32 + ctx 0 gen 256, warm 1 + 3;
#           then --profile-ops with pq.py 8k x 2), rounds B T T B -> W/res/perf/
#   df2     the drafter (RULES §3): servers P3-mx and P3t4 (drafter on), P3t4 spec 0; gen-prompts (16) x 256 greedy at C 1
#           (w155's st_acc), /metrics draft counters, GET /graph notes; dcmp / tiemargin -> W/res/df2/
set -u
Q=/data/q27 W=/data/q27/w157 C=1 L=/data/q27/locks I=/data/q27/rad10-perf/install HD=$W/b10e1/home
G=$W/in/golden-s173e R=$W/res; mkdir -p $R/gt $R/perf $R/df2 $W/logs; ulimit -n 65536
say() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" >> $W/logs/gpu157.log; }
gpu() { ( flock 5; flock -s 4; "$@" ) 5>$L/card$C.lock 4>$L/ram.lock; }
libs() { case $1 in 10) echo /opt/rocm-10.1/lib ;; 714) S=$Q/venv/lib/python3.14/site-packages; echo $S/_rocm_sdk_core/lib:$S/_rocm_sdk_libraries/lib:$S/_rocm_sdk_devel/lib ;; esac; }
st_gtest() {
  for tc in 10 714; do
    local T=$W/b${tc}e1/pb/exl3dense/exl3dense_embed_exl3i_gpu_test
    for a in dev host; do
      local o=$R/gt/${tc}e1-embed-$a.txt; [ -f $o ] && grep -q '^rc 0' $o && continue
      X=(); [ $a = host ] && X=(--host)
      gpu bash -c "HIP_VISIBLE_DEVICES=$C LD_LIBRARY_PATH=$(libs $tc) timeout 1800 $T ${X[*]} > $o 2>&1; echo rc \$? >> $o"
      say "gtest $tc embed $a: $(tail -n 2 $o | tr '\n' ' ')"
    done
    local o=$R/gt/${tc}e1-ctall.txt; [ -f $o ] && grep -q '^done' $o && continue
    bash $W/tools/ctall.sh $C $tc e1        # gt.sh takes the card lock (+ ram shared) per test
    say "gtest $tc ctall: $(grep -c 'PASS' $o) PASS lines, $(grep -c -i 'fail' $o) fail lines"
  done
}
st_kld() {
  for a in "$@"; do
    local o=$R/kld-$a; [ -f $o.json.rows ] && continue
    [ -f $W/mix/$a.rad ] || { say "kld $a: no mix"; continue; }
    gpu bash $Q/tools/score.sh $I $HD $W/mix/$a.rad $G $o $C > /dev/null
    say "kld $a: $(grep -E 'I kld: [0-9]+ / ' $o.log | tail -n 1) | $(grep -m1 'static weights' $o.log | tr -s ' ') | rows $(sha256sum < $o.json.rows 2>/dev/null | cut -c1-16)"
  done
}
PORT=$((8570 + C)); SP=
up() {  # up LOG MODEL ENV [radiance args]
  local lg=$1 m=$2 e=$3 i=0; shift 3
  env $e ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$HD $I/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT "$@" > $lg 2>&1 &
  SP=$!
  while [ $i -lt 1800 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
  curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null
}
down() { kill -INT $SP 2>/dev/null; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null; SP=; }
perf1() {  # perf1 ROW ROUND
  local row=$1 r=$2 P=$R/perf T=$Q/w138/tools PQ=$Q/w137/pq.py PR=$Q/w137/prompts
  for mode in e2e prof; do
    local t=$mode-$row-r$r; [ -f $P/$t.done ] && continue
    X=(); [ $mode = prof ] && X=(--profile-ops)
    if up $P/serve-$t.log $W/mix/$row.rad "" "${X[@]}"; then
      if [ $mode = prof ]; then python3 -I $PQ $PORT $PR/ctx-8k.txt 2 > $P/bench-$t.txt 2>&1
      else
        python3 -I $PQ $PORT $PR/ctx-8k.txt 4 > $P/bench-$t.txt 2>&1
        python3 -I $T/bench.py --port $PORT --prompts $PR --out $P/$t.jsonl.part --tag $t --ctx 8k --warm 1 --runs 3 --gen 32 \
          --server-log $P/serve-$t.log >> $P/bench-$t.txt 2>&1 &&
        python3 -I $T/bench.py --port $PORT --prompts $PR --out $P/$t.jsonl.part --tag $t --ctx 0 --warm 1 --runs 3 --gen 256 \
          --server-log $P/serve-$t.log >> $P/bench-$t.txt 2>&1 && mv $P/$t.jsonl.part $P/$t.jsonl
      fi
      down
      if [ $mode = prof ]; then grep -q 'per-op timing' $P/serve-$t.log && touch $P/$t.done; else [ -f $P/$t.jsonl ] && touch $P/$t.done; fi
      say "perf $t: $(grep -m1 'static weights' $P/serve-$t.log | tr -s ' ') | $(tail -n 1 $P/bench-$t.txt)"
    else down; say "perf $t: no health ($(grep -m1 '^E ' $P/serve-$t.log | cut -c1-200))"; fi
  done
}
st_perf() { for x in B:1 T:1 T:2 B:2; do gpu perf1 ${x%:*} ${x#*:}; done; }
df1() {  # df1 TAG MODEL ENV [args]
  local t=$1 m=$2 e=$3 D=$R/df2 TL=$W/in/tools; shift 3
  [ -f $D/$t.done ] && return 0
  if up $D/serve-$t.log $m "$e" "$@"; then
    curl -s -m 60 http://127.0.0.1:$PORT/graph > $D/graph-$t.json
    python3 -I $TL/gen.py $PORT $W/in/prompts/gen-prompts.jsonl $D/txt-$t.jsonl 1 256 > /dev/null 2>&1
    curl -s -m 10 http://127.0.0.1:$PORT/metrics > $D/metrics-$t.txt
    down; touch $D/$t.done
    say "df2 $t: $(grep -c '"error"' $D/txt-$t.jsonl) errors; $(grep -E '^radiance:draft_(tokens|accepted)_total' $D/metrics-$t.txt | awk '{printf "%s ", $2}'); $(grep -m1 -o 'block embedded by [a-z_0-9]*' $D/serve-$t.log)"
  else down; say "df2 $t: no health ($(grep -m1 -E '^E |refus' $D/serve-$t.log | cut -c1-200))"; fi
}
st_df2() {
  gpu df1 p3-on $W/in/P3-mx.rad ""
  gpu df1 t4-on $W/mix/P3t4.rad ""
  gpu df1 t4-off $W/mix/P3t4.rad "" --num-speculative-tokens 0
  gpu df1 p3-off $W/in/P3-mx.rad "" --num-speculative-tokens 0
  local D=$R/df2 TL=$W/in/tools
  for x in t4-off:t4-on p3-off:p3-on p3-on:t4-on; do
    a=${x%:*}; b=${x#*:}
    python3 -I $TL/dcmp.py $D/txt-$a.jsonl $D/txt-$b.jsonl > $D/dcmp-$a-vs-$b.txt 2>&1
    python3 -I $TL/tiemargin.py $D/txt-$a.jsonl $D/txt-$b.jsonl > $D/tie-$a-vs-$b.txt 2>&1
  done
  say "df2 cmp done"
}
say "start $* (pid $$)"
case $1 in
  gtest) st_gtest ;; kld) shift; st_kld "$@" ;; perf) st_perf ;; df2) st_df2 ;;
  *) echo "stage?"; exit 2 ;;
esac
say "end $1"
