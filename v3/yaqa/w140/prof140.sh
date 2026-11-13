#!/bin/bash
# prof140.sh -- #162-02 (w140): head GEMV time, mul1 vs i8 at equal K, from radiance --profile-ops (the whole-model decode
# step of r16m140.sh moves ±7% between rounds on a shared ds8, too much for RULES' 1% bar). Per pack (r16m140.sh's packs
# i6 m6 i6.5 m6.5, ROUNDS interleaved): serve with --profile-ops on CARD, one greedy completion (short prompt, 512 tokens,
# ignore_eos), stop the server (SIGTERM; it prints the per-op table at exit, as run176.sh); headops140.py sums the output.weight / logits rows.
# Locks: card + ram shared. Restartable (res/prof-TAG-ROUND.done).
set -u; Q=/data/q27; O=$Q/rad10-perf; W=$Q/heads/w140; L=$Q/locks; PR=$Q/w124/prompts
CARD=${CARD:-6}; PORT=${PORT:-8141}; ROUNDS=${ROUNDS:-"1 2"}; HEADS=${HEADS:-"i6 m6 i6.5 m6.5"}
P=$W/packs; R=$W/res; exec >> $W/prof140.log 2>&1; ulimit -n 65536
say() { echo "$(date -u +%T) $*"; }
prof() {   # prof HEAD ROUND
  local h=$1 r=$2 t=$1-$2 i=0; [ -f $R/prof-$t.done ] && return 0; [ -f $P/$h.rad ] || { say "prof $t: no pack"; return 0; }
  ( flock 5; flock -s 6; say "prof $t (card $CARD)"
    ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$O/home $O/install/bin/radiance --model $P/$h.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 12288 --max-num-seqs 8 --host 127.0.0.1 --port $PORT --profile-ops > $R/serve-prof-$t.log 2>&1 &
    SP=$!
    while [ $i -lt 900 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 2; i=$((i + 2)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      python3 -c "import json,sys; print(json.dumps({'model':'q','prompt':open(sys.argv[1]).read(),'max_tokens':512,'temperature':0,'ignore_eos':True}))" \
        $PR/short.txt | curl -s -m 600 -o $R/prof-$t.json http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' -d @-
      kill $SP 2>/dev/null; wait $SP 2>/dev/null
      grep -q 'per-op timing' $R/serve-prof-$t.log && touch $R/prof-$t.done
      say "prof $t: $(grep -c . $R/serve-prof-$t.log) log lines, table $([ -f $R/prof-$t.done ] && echo yes || echo no)"
    else kill $SP 2>/dev/null; wait $SP 2>/dev/null; say "prof $t: no health"; fi
  ) 5>$L/card$CARD.lock 6>$L/ram.lock
}
for r in $ROUNDS; do for h in $HEADS; do [ -f $W/STOP ] && exit 0; prof $h $r; done; done
say "PROF140 DONE"
