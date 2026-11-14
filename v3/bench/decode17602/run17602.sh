#!/bin/bash
# run17602.sh CARD LIST [LIST ...] -- #176-02 (w159) decode speed table from e2e: one radiance server session per row, as a
# user runs it (qc-exl3-v3 120132a, home rad10-e157b; --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 --max-num-seqs 8),
# **f16 decode activations** (user default: QWEN35_EXL3_AQ_XB=0 EXL3I_MAD_GEMV=0, i.e. nt_pass on f16 rows; prefill A8
# where the pack has act lines), w138's bench.py over the OpenAI chat API: batch 1, greedy, ctx 0 (warm 1 + 3 x 256 tokens)
# and, per row, ctx 8k (warm 1 + 2 x 128 tokens). Server ms per decode step = the engine's decode-step seconds / steps.
# LIST lines: TAG PACK [OPT ...]; OPT: stamp (RADIANCE_AQL_STAMP=8: device times of every dispatch of every 8th pass ->
# res/st-TAG.json via st17602.py; the CSV itself is deleted), ctx=0,8k (default 0,8k), env=K=V,K=V (extra server env),
# wait (the pack is still being built by mk17602.sh: wait for it, or skip the row if PACK.fail appears).
# Rows run in file order; finished rows (res/TAG.jsonl) are skipped, so a rerun resumes. The card's lock is held for one
# row at a time (+ ram.lock shared); rocm-smi snapshots before / after each row go into res/smi-TAG.json.
# QUIET=1 (env): the e2e rounds. W/QUIET exists from the first row to the end of the lists (mk17602.sh starts no new
# build while it does: radmix / rad-convert on the host spike the step time), and each row holds W/quiet.lock exclusive
# (a build in flight holds it shared, so the first row waits for it to end).
# Logs: W/logs/run.log. Stop: touch W/STOP (checked between rows). (w159)
set -u
C=$1; shift
Q=/data/q27; W=$Q/w159; R=$W/res; L=$Q/locks; T=$W/tools; PORT=$((18760 + C))
I=${RAD_I:-$Q/rad10-e157b/install}; H=${RAD_H:-$Q/rad10-e157b/home}
SI=$((6 - 2 * (C / 2) + C % 2))                     # ds8: HIP i <-> rocm-smi GPU[6 - 2 floor(i/2) + i mod 2]
mkdir -p $R $W/logs; exec >> $W/logs/run.log 2>&1; ulimit -n 65536
say() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S') [hip $C] $*"; }
smi() { rocm-smi -d $SI --showtemp --showclocks --showpower --showmaxpower --showuse --showmeminfo vram --json 2>/dev/null; }

row() {   # TAG PACK OPTS...
  local tag=$1 pack=$2; shift 2
  local stamp=0 ctxs=0,8k ev="" o
  local wt=0
  for o in "$@"; do case $o in stamp) stamp=1;; ctx=*) ctxs=${o#ctx=};; env=*) ev=${o#env=};; wait) wt=1;; esac; done
  [ -f $R/$tag.jsonl ] && return 0
  if [ $wt = 1 ]; then
    while [ ! -f $pack ] && [ ! -f $pack.fail ] && [ ! -f $W/STOP ]; do sleep 15; done
  fi
  [ -f $pack ] || { say "$tag: no pack $pack (skipped)"; return 0; }
  (
    [ "${QUIET:-0}" = 1 ] && flock 7
    flock 5; flock -s 6
    local sv="" st=$R/st-$tag.csv
    [ $stamp = 1 ] && { rm -f $st; sv="RADIANCE_AQL_STAMP=8 RADIANCE_AQL_STAMP_OUT=$st"; }
    echo "{\"before\": $(smi || echo null)," > $R/smi-$tag.json
    say "$tag: serve $pack (stamp $stamp, ctx $ctxs, env '$ev')"; t0=$(date +%s)
    env QWEN35_EXL3_AQ_XB=0 EXL3I_MAD_GEMV=0 ${ev//,/ } $sv ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance \
      --model $pack --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 --max-num-seqs 8 --host 127.0.0.1 --port $PORT \
      > $R/serve-$tag.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 1200 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 2; i=$((i + 2)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      t1=$(date +%s)
      meta="{\"row\":\"$tag\",\"home\":\"$H\",\"install\":\"$I\",\"pack\":\"$pack\",\"env\":\"QWEN35_EXL3_AQ_XB=0,EXL3I_MAD_GEMV=0${ev:+,$ev}\",\"stamp\":$stamp,\"card_hip\":$C,\"load_s\":$((t1 - t0))}"
      rm -f $R/$tag.jsonl.part; ok=1
      for c in ${ctxs//,/ }; do
        if [ $c = 0 ]; then a="--warm 1 --runs 3 --gen 256"; else a="--warm 1 --runs 2 --gen 128"; fi
        python3 -I $T/bench.py --port $PORT --prompts $W/prompts --out $R/$tag.jsonl.part --tag $tag --ctx $c $a \
          --server-log $R/serve-$tag.log --meta "$meta" >> $R/bench-$tag.txt 2>&1 || { ok=0; break; }
      done
      kill -INT $SP; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
      echo "\"after\": $(smi || echo null)}" >> $R/smi-$tag.json
      if [ $ok = 1 ]; then
        if [ $stamp = 1 ]; then
          python3 -I $T/st17602.py $st > $R/st-$tag.json.part 2> $R/st-$tag.err && mv $R/st-$tag.json.part $R/st-$tag.json && rm -f $st
        fi
        mv $R/$tag.jsonl.part $R/$tag.jsonl
        say "$tag: done in $(( $(date +%s) - t0 )) s (load $((t1 - t0)) s): $(grep -o 'server [0-9.]* ms/step' $R/bench-$tag.txt | tail -n 4 | tr '\n' ' ')"
      else say "$tag: bench FAILED ($(tail -n 1 $R/bench-$tag.txt))"; fi
    else
      kill $SP 2>/dev/null; wait $SP 2>/dev/null
      say "$tag: no health ($(grep -m1 '^E ' $R/serve-$tag.log | cut -c1-200))"
    fi
  ) 5>$L/card$C.lock 6>$L/ram.lock 7>$W/quiet.lock < /dev/null
}

[ "${QUIET:-0}" = 1 ] && { touch $W/QUIET; trap 'rm -f $W/QUIET' EXIT; }
for f in "$@"; do
  while read -r tag pack opts; do
    case $tag in ''|\#*) continue;; esac
    [ -f $W/STOP ] && { say "STOP"; exit 0; }
    row $tag $pack $opts
  done < $f
done
say "LISTS DONE: $*"
