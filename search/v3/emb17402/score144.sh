#!/bin/bash
# score144.sh GOLDEN NAME... -- (w144, #174-02, box2) KLD of W/mix/NAME.rad on GOLDEN (box2 tools/score.sh: tp 1, bf16 KV,
# max-model-len 16384, one sequence) with rad10-perf's radiance and W/home (this branch's plugins), one after another on
# the first free card (polling the card locks, as quant144.sh), ram.lock shared. Out
# W/scores/NAME-<golden>.json(.rows,.log); skips finished ones. Prints the load's VRAM / placement lines and the KLD line.
set -u
Q=/data/q27 W=/data/q27/w144 LK=/data/q27/locks G=$1; shift
gn=$(basename $G); mkdir -p $W/scores $W/run
todo=(); for n in "$@"; do [ -f $W/scores/$n-$gn.json.rows ] || todo+=($n); done
[ ${#todo[@]} = 0 ] && { echo "nothing to score"; exit 0; }
while :; do                          # the first free card by polling its lock, held on fd 9 until the end
  c=
  for cc in 0 1 2 3; do exec 9>$LK/card$cc.lock; if flock -n 9; then c=$cc; break; fi; exec 9>&-; done
  [ -n "$c" ] && break; sleep 2
done
for n in "${todo[@]}"; do
  o=$W/scores/$n-$gn
  flock -s $LK/ram.lock bash $Q/tools/score.sh $Q/rad10-perf/install $W/home $W/mix/$n.rad $G $o $c
  echo "$n card $c: $(grep -E 'static weights|placement: .token_embd|I kld: [0-9]+ / ' $o.log | tr '\n' '|')"
done
exec 9>&-
