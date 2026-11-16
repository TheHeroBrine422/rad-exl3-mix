#!/bin/bash
# speed178.sh ROWS -- w178 (#194-02): decode and prefill (TTFT) of the KL-IQR entries with two-pass activations OFF
# (RADIANCE_ACT_TWO=0, a load-time override) and as a same-day control (no override). Same harness as w177's speed.sh:
# one radiance serve per row on box2 card 0 (CARD, HIP index; default 0 = rocm-smi GPU[3]), bench.py over the OpenAI chat
# API (batch 1, greedy, ignore_eos, 512 tokens, contexts 0 / 4k / 16k, 2 warm-ups, 5 runs), --tp 1 --kv-cache-dtype bf16
# --max-model-len 40960 --max-num-seqs 8, speculative decoding off. Card lock per row, ram lock shared, rocm-smi snapshots
# before / loaded / after. Restartable: a row with a finished <TAG>.done is skipped.
# ROWS lines: TAG PACKKEY ENV(VAR=VALUE or -). Models and build: w177's (read only). Outputs: /data/q27/w178/speed/.
set -u; W=/data/q27/w178; C=${CARD:-0}; SI=$((3 - C)); PORT=${PORT:-8378}; Q=/data/q27; L=$Q/locks; PY=$Q/venv/bin/python
R=$W/speed; T=$W/tools; mkdir -p $R $W/logs; ulimit -n 65536
exec >> $W/logs/speed.log 2>&1
say() { echo "$(date -u +%T) $*"; }
BIN=$Q/w177/install/bin/radiance; RH=$Q/w177/install/share/radiance
declare -A PK=(
  [nano]=$Q/w177/models/nano/qwen3.8-27b-kl-nano.rad [tiny]=$Q/w177/models/tiny/qwen3.8-27b-kl-tiny.rad [small]=$Q/w177/models/small/qwen3.8-27b-kl-small.rad)
smi() { rocm-smi -d $SI --showtemp --showclocks --showpower --showmaxpower --showuse --showmeminfo vram --json 2>/dev/null; }
pids() { rocm-smi --showpids 2>/dev/null | awk -v g=$C '$3 == g'; }
up() {   # up TAG PACK [env...]: start the server, wait for /health
  local t=$1 m=$2 i=0; shift 2
  env ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$RH "$@" $BIN --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 \
    --max-num-seqs 8 --host 127.0.0.1 --port $PORT > $R/serve-$t.log 2>&1 &
  SP=$!; T0=$(date +%s.%N)
  while [ $i -lt 900 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 1; i=$((i + 1)); kill -0 $SP 2>/dev/null || break; done
  curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && LOADS=$(awk -v a=$(date +%s.%N) -v b=$T0 "BEGIN{printf \"%.1f\", a - b}")
}
down() { kill $SP 2>/dev/null; wait $SP 2>/dev/null; }
sampler() { while :; do echo "{\"t\":$(date +%s.%N),\"load\":\"$(cut -d' ' -f1-3 /proc/loadavg)\",\"smi\":$(smi)}"; sleep 2; done; }
row() {   # row TAG PACKKEY ENV
  local t=$1 k=$2 e=$3 m=${PK[$2]} envs=()
  [ -f $R/$t.done ] && return 0
  [ -f $m ] || { say "row $t: pack missing ($m)"; return 0; }
  [ "$e" = "-" ] || envs=("$e")
  ( flock 5; flock -s 6
    say "row $t: $(basename $m) env [${envs[*]:-none}]"
    rm -f $R/$t.jsonl.part
    echo "{\"pre\":$(smi),\"pids\":\"$(pids | tr -s ' \t' ' ')\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $R/smi-$t-pre.json
    if up $t $m "${envs[@]}"; then
      echo "{\"loaded\":$(smi),\"load_s\":$LOADS}" > $R/smi-$t-loaded.json
      sampler > $R/smi-$t.jsonl & SMP=$!
      meta=$(printf '{"build":"%s","bin":"%s","home":"%s","pack":"%s","pack_bytes":%s,"spec":"off","env":"%s","args":"","card_hip":%s,"load_s":%s}' \
        w177-theirs $BIN $RH $(basename $m) $(stat -c %s $m) "${envs[*]:-none}" $C "${LOADS:-0}")
      $PY -I $T/bench.py --port $PORT --prompts /data/q27/w177/prompts --out $R/$t.jsonl.part --tag $t --server-log $R/serve-$t.log \
        --ctx 0,4k,16k --warm 2 --runs 5 --gen 512 --meta "$meta" > $R/bench-$t.txt 2>&1 && mv $R/$t.jsonl.part $R/$t.jsonl
      kill $SMP; wait $SMP 2>/dev/null
      echo "{\"post\":$(smi),\"pids\":\"$(pids | tr -s ' \t' ' ')\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $R/smi-$t-post.json
      down
      [ -f $R/$t.jsonl ] && touch $R/$t.done
      say "row $t: $(tail -n 1 $R/bench-$t.txt | cut -c1-300)"
    else down; say "row $t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-200))"; fi
  ) 5>$L/card$C.lock 6>$L/ram.lock
}
grep -v '^#' $T/rows.txt | while read -r t k e; do [ -n "$t" ] && row $t $k "$e"; done
say "SPEED DONE"
