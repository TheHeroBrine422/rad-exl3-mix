#!/bin/bash
# qad182.sh -- (ds8, CPU + ram.lock shared) #182: w154's #181 QAD containers (the #171 2.5 incumbent in served form:
# ctl unhealed, a1m / a4m = arm A's decayed 1 M / 4 M checkpoints, p1m = 1 M calib-only; text-only) + the bf16 tower
# (radmix add v.*, packs/tower.mix) -> W/packs/Q-*.rad, listed in W/cands-qad.txt. Restartable. (w156)
W=/data/q27/w156; Q=/data/q27; P=$W/packs; L=$Q/locks; S=$Q/w154/eval/inc2.5
: > $W/cands-qad.txt
for a in ctl a1m a4m p1m; do
  if [ ! -f $P/Q-$a.rad ] && [ -f $S/$a/mix.rad ]; then
    ( flock -s 6; $W/tools/radmix $P/Q-$a.rad.part $S/$a/mix.rad $P/tower.mix > $P/Q-$a.mix.log 2>&1 && mv $P/Q-$a.rad.part $P/Q-$a.rad ) 6>$L/ram.lock
    echo "$(date -u +%T) Q-$a: $(tail -n 1 $P/Q-$a.mix.log)"
  fi
  [ -f $P/Q-$a.rad ] && echo "Q-$a $P/Q-$a.rad" >> $W/cands-qad.txt
done
