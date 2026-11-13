#!/bin/bash
# r16check.sh -- #164-02 (w136): the R16 head as a real quantized head through the staged rad10-perf build (box2).
# Per head (R16s003 K4 = the search's u4, K2.5 = u2.5; codebook i8, rows in R16 order, ourq --head-perm R16.npy):
#   A<h> = rad-convert (rad10-perf/home-pack, --no-error) of the ladder K4 trunk recipe + the head's exl3i line (rows=r16)
#   B<h> = radmix of A<h> with output.weight from the search's bf16 donor hd-u<h>.rad (headslab.py: the same R16 head
#          dequantized and un-permuted to vocab order): same container, the head spliced as bf16 (what #171 scores)
# then radents.py (A vs B: only output.weight differs; the head's stored bytes vs the menu's), KLD on golden-ver3-8k
# (rad10-perf install + home, bf16 KV) and kldpair.py, and served A / B interleaved x 2 on one card: ask.sh (text, top-3
# logprobs, prefill 8k x 3) and dstep.sh (decode step, C 1, short and 8k prompts). Locks: conv + ram exclusive per pack,
# card + ram shared per eval. Restartable (skips finished outputs). CARD (default 0), CARD2 (default 1: the K2.5 KLDs).
# ROUNDS="3 4 5": serve rounds only (A4 B4 A2.5 B2.5 interleaved), for a timing pass with nothing else running.
set -u; Q=/data/q27; O=$Q/rad10-perf; W=$Q/w136; L=$Q/locks; G=$Q/goldens/golden-ver3-8k; PY=$Q/venv/bin/python
T=$Q/w124/tools; PR=$Q/w124/prompts; S=/data/q27/tools/rad10perf; CARD=${CARD:-0}; CARD2=${CARD2:-1}; PORT=${PORT:-8136}
P=$W/packs; R=$W/res; mkdir -p $P $R $W/recipes; exec >> $W/r16check.log 2>&1; ulimit -n 65536
say() { echo "$(date -u +%T) $*"; }
LV=$W/ladder/q27v3-144dd6e1-k4-i8
sed "s#src=$Q/ladder/#src=$W/ladder/#" $LV/final.recipe > $W/recipes/K4.recipe
for h in 4 2.5; do
  { cat $W/recipes/K4.recipe; echo "output.weight exl3i src=$W/heads/q/R16s003/K$h codebook=mul1i8 rate=$h rows=r16"; } > $W/recipes/A$h.recipe
  echo "output.weight $W/heads/donors/hd-u$h.rad" > $W/recipes/B$h.mix
done
pack() {   # pack NAME
  local n=$1; [ -f $P/$n.rad ] && return 0
  rm -f $P/$n.rad.part; say "pack $n: waiting for conv.lock + ram.lock"
  ( flock -x 7; flock -x 6; say "pack $n: start"; t0=$(date +%s)
    cd $Q && RADIANCE_HOME=$O/home-pack RADIANCE_HOST_DEVICES=1 nice -n 10 $O/install/bin/rad-convert --no-error models/qwen38-27b-bf16 \
      --recipe $W/recipes/$n.recipe --set quantization_config.quant_method=exl3 -o $P/$n.rad.part > $R/pack-$n.log 2>&1 && mv $P/$n.rad.part $P/$n.rad
    say "pack $n: rc $? $(stat -c %s $P/$n.rad 2>/dev/null) bytes, $(( $(date +%s) - t0 )) s"
  ) 7>$L/conv.lock 6>$L/ram.lock
}
mix() {   # mix NAME BASE
  local n=$1 b=$2; [ -f $P/$n.rad ] && return 0; [ -f $P/$b.rad ] || return 0
  ( flock -s 6; $Q/tools/radmix $P/$n.rad.part $P/$b.rad $W/recipes/$n.mix > $R/mix-$n.log 2>&1 && mv $P/$n.rad.part $P/$n.rad
    say "mix $n: $(tail -n 1 $R/mix-$n.log)" ) 6>$L/ram.lock
}
ents() { [ -f $R/ents-$1.json ] || $PY -I $S/radents.py $P/A$1.rad $P/B$1.rad > $R/ents-$1.json; say "ents $1: $(cat $R/ents-$1.json | cut -c1-400)"; }
kld() {   # kld TAG CARD
  local t=$1 c=$2; [ -f $R/kld-$t.json ] && return 0; [ -f $P/$t.rad ] || { say "kld $t: no pack"; return 0; }
  ( flock 5; flock -s 6; say "kld $t (card $c)"
    ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$O/home timeout 3600 $O/install/bin/radiance --model $P/$t.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-$t.json.part > $R/kld-$t.log 2>&1 \
      && mv $R/kld-$t.json.part.rows $R/kld-$t.json.rows && mv $R/kld-$t.json.part $R/kld-$t.json
    say "kld $t rc $? $($PY -c "import json; j=json.load(open('$R/kld-$t.json'))['all']; print(j['kld']['mean'], j['top1_agreement'])" 2>/dev/null)"
  ) 5>$L/card$c.lock 6>$L/ram.lock
}
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
pack A4; mix B4 A4; pack A2.5; mix B2.5 A2.5
for h in 4 2.5; do [ -f $P/B$h.rad ] && ents $h; done
( kld A2.5 $CARD2; kld B2.5 $CARD2 ) &
kld A4 $CARD; kld B4 $CARD; wait
$PY -I $S/kldpair.py $G $R A4 B4 A2.5 B2.5 > $R/kldpair.json 2> $R/kldpair.err; say "kldpair: $(cat $R/kldpair.json $R/kldpair.err)"
if [ -z "${ROUNDS:-}" ]; then
  for r in 1 2; do serve A4-$r; serve B4-$r; done
  for r in 1 2; do serve A2.5-$r; serve B2.5-$r; done
else   # ROUNDS="3 4 5": the timing pass alone (nothing else on the box), the four packs interleaved a round
  for r in $ROUNDS; do serve A4-$r; serve B4-$r; serve A2.5-$r; serve B2.5-$r; done
fi
say "R16CHECK DONE"
