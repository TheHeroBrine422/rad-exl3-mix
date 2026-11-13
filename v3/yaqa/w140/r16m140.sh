#!/bin/bash
# r16m140.sh -- #162-02 (w140, RULES.md): mul1 vs i8 R16 heads through the staged rad10-perf build on ds8 (r16check.sh's
# procedure, w136). Per head h in i6 m6 i6.5 m6.5 m7.5 m8 (i = R16s003 i8 -> codebook=mul1i8, m = R16s003m1 -> mul1):
#   P<h> = rad-convert (rad10-perf/home-pack, --no-error) of the ladder K4 trunk recipe + the head's exl3i line (rows=r16)
# then radents.py (head's stored bytes) and served one card, packs interleaved per round: ask.sh (text, top-3 logprobs)
# and dstep.sh (decode step, C 1, short and 8k prompts). Locks: conv exclusive + ram shared per pack (ds8: 235 GB),
# card + ram shared per serve. Restartable (skips finished outputs). CARD (default 6), ROUNDS (default "1 2 3"), SERVE
# (heads to serve, default all; run: ROUNDS=1 for all six, then ROUNDS="2 3" SERVE="i6 m6 i6.5 m6.5": the equal-K timing).
set -u; Q=/data/q27; O=$Q/rad10-perf; W=$Q/heads/w140; L=$Q/locks; PY=$Q/venv/bin/python
T=$Q/w124/tools; PR=$Q/w124/prompts; S=${S:-$Q/heads/w140/code}; CARD=${CARD:-6}; PORT=${PORT:-8140}; ROUNDS=${ROUNDS:-"1 2 3"}
P=$W/packs; R=$W/res; mkdir -p $P $R $W/recipes; exec >> $W/r16m140.log 2>&1; ulimit -n 65536
say() { echo "$(date -u +%T) $*"; }
HEADS=${HEADS:-"i6 m6 i6.5 m6.5 m7.5 m8"}   # amendment 1: HEADS="i4 i4.125 i4.25 i4.5"
for h in $HEADS; do
  k=${h:1}; [ ${h:0:1} = i ] && { d=R16s003; cb=mul1i8; } || { d=R16s003m1; cb=mul1; }
  { cat $Q/ladder/q27v3-144dd6e1-k4-i8/final.recipe; echo "output.weight exl3i src=$Q/heads/q/$d/K$k codebook=$cb rate=$k rows=r16"; } > $W/recipes/$h.recipe
done
pack() {   # pack NAME
  local n=$1; [ -f $P/$n.rad ] && return 0
  rm -f $P/$n.rad.part; say "pack $n: waiting for conv.lock + ram.lock"
  ( flock -x 7; flock -s 6; say "pack $n: start"; t0=$(date +%s)
    cd $Q && RADIANCE_HOME=$O/home-pack RADIANCE_HOST_DEVICES=1 nice -n 10 $O/install/bin/rad-convert --no-error models/qwen38-27b-bf16 \
      --recipe $W/recipes/$n.recipe --set quantization_config.quant_method=exl3 -o $P/$n.rad.part > $R/pack-$n.log 2>&1 && mv $P/$n.rad.part $P/$n.rad
    say "pack $n: rc $? $(du -sb $P/$n.rad 2>/dev/null | cut -f1) bytes, $(( $(date +%s) - t0 )) s"
  ) 7>$L/conv.lock 6>$L/ram.lock
}
ents() { [ -s $R/ents-$1.json ] || $PY -I $S/radents.py $P/$1.rad > $R/ents-$1.json; say "ents $1: $(grep -o "\"output.weight\"[^]]*]" $R/ents-$1.json | head -1)"; }
up() {   # up TAG PACK
  local t=$1 m=$2 i=0
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$O/home $O/install/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 12288 \
    --max-num-seqs 8 --host 127.0.0.1 --port $PORT > $R/serve-$t.log 2>&1 &
  SP=$!
  while [ $i -lt 900 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 2; i=$((i + 2)); kill -0 $SP 2>/dev/null || break; done
  curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null
}
down() { kill $SP 2>/dev/null; wait $SP 2>/dev/null; }
serve() {   # serve TAG-ROUND
  local t=$1 n=${1%-*}; [ -f $R/s-$t.done ] && return 0; [ -f $P/$n.rad ] || { say "serve $t: no pack"; return 0; }
  ( flock 5; flock -s 6; say "serve $t (card $CARD)"
    if up $t $P/$n.rad; then
      bash $T/ask.sh $PORT ctx8k > $R/ask-$t.txt 2>&1
      P=$PORT PROMPT=$PR/short.txt LEVELS=1 bash $T/dstep.sh > $R/dstep-$t-short.txt 2>&1
      P=$PORT PROMPT=$PR/ctx8k.txt LEVELS=1 bash $T/dstep.sh > $R/dstep-$t-8k.txt 2>&1
      touch $R/s-$t.done; say "serve $t: $(grep -h '^1 ' $R/dstep-$t-short.txt $R/dstep-$t-8k.txt | awk '{print $5}' | tr '\n' ' ')ms/step"
    else say "serve $t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-160))"; fi
    down ) 5>$L/card$CARD.lock 6>$L/ram.lock
}
say "start (rad10-perf $(head -n 1 $O/src/COMMIT))"
for h in $HEADS; do [ -f $W/STOP ] && exit 0; pack $h; [ -f $P/$h.rad ] && ents $h; done
for r in $ROUNDS; do for h in ${SERVE:-$HEADS}; do [ -f $W/STOP ] && exit 0; serve $h-$r; done; done
say "R16M140 DONE"
