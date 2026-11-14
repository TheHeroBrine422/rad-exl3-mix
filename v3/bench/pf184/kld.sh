#!/bin/bash
# kld.sh CARD TAG=HOME=PACK[=ENV]... -- whole-model KLD rows (box2 tools/score.sh settings: tp 1, bf16 KV, one sequence;
# golden-search27c-c unless GOLD is set: an equivalence check of kernel builds, no selection) for each TAG with the
# rad10-e157b install (this branch's base core) and plugin HOME ("-": rad10-e157b's own), under card CARD's lock and
# ram.lock shared; ENV "K=V,K=V" for the server. Out res/kld/TAG.json(.rows); skips finished. (w150 kld150.sh, #184 w160)
C=$1; shift; Q=/data/q27; W=$Q/w160; G=${GOLD:-$Q/goldens/golden-search27c-c}; I=$Q/rad10-e157b/install; mkdir -p $W/res/kld
for th in "$@"; do
  IFS='=' read -r t H P EV <<< "$th"; [ "$H" = - ] && H=$Q/rad10-e157b/home; o=$W/res/kld/$t
  [ -f $o.json ] && { echo "$t: done"; continue; }
  flock -s $Q/locks/ram.lock flock $Q/locks/card$C.lock timeout 1800 env ${EV//,/ } bash $Q/tools/score.sh $I $H $P $G $o $C
done
