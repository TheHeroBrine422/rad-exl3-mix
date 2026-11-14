#!/bin/bash
# kld150.sh CARD TAG=HOME=PACK... -- whole-model KLD rows (tools/score.sh: rad10-perf install, bf16 KV, golden
# search27c-c unless GOLD is set) for each TAG under card CARD's lock and ram.lock shared; out res/kld/TAG.json(.rows);
# skips finished. (w150, #177-02)
C=$1; shift; Q=/data/q27; W=$Q/w150; G=${GOLD:-$Q/goldens/golden-search27c-c}; I=$Q/rad10-perf/install; mkdir -p $W/res/kld
for th in "$@"; do
  t=${th%%=*}; r=${th#*=}; H=${r%%=*}; P=${r#*=}; o=$W/res/kld/$t
  [ -f $o.json ] && { echo "$t: done"; continue; }
  flock -s $Q/locks/ram.lock flock $Q/locks/card$C.lock timeout 1800 bash $Q/tools/score.sh $I $H $P $G $o $C
done
