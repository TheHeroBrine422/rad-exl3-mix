#!/bin/bash
# moves.sh CARD NAME... -- radmix (timed, ext4) + score on golden-search27 for each move; restartable (skips finished)
cd /data/q27/w123; C=$1; shift
B=/data/q27/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; G=/data/q27/w123/gold/golden-search27
for n in "$@"; do
  [ -f scores/s27-$n.json.rows ] && continue
  if [ "$n" = base ]; then M=$B; else
    M=mixes/mv-$n.rad; rm -f $M
    t=$(date +%s.%N); ./mixes/radmix $M.part $B mixfiles/mv-$n.mix > logs/radmix-$n.log 2>&1 && mv $M.part $M; sync -f $M; t1=$(date +%s.%N)
    echo "{\"radmix\": \"$n\", \"s\": $(python3 -c "print(round($t1-$t,2))"), \"bytes\": $(stat -c %s $M)}" >> logs/radmix.jsonl
  fi
  ./run.sh $C s27-$n $M $G >> logs/scores.jsonl; echo "$n rc=$?"
  [ "$n" != base ] && rm -f $M          # ext4: a mix is a full copy (no reflinks); the .mix file rebuilds it
done
