#!/bin/bash
# watch181.sh SPEC... (w154, #181; runs on ds8) -- the judgement pipeline for checkpoints as they appear. SPEC =
# MIX:TAG:RUN/CKPT (e.g. inc2.5:a4m:inc2.5g/ckpt-4e+06). For each: wait for RUN/CKPT/.done on box2, build the healed mix
# there (eval181.py without slices: apply scales, pack, mix, census; CPU, conv + ram locks), pull mix.rad over the 10 GbE
# link, score c + x181g here on the first free card of $CARDS (flock -n; never queues on a held card). Restartable.
set -u
cd /data/q27/w154 && source env.sh
CARDS=${CARDS:-7}
B="ssh -i /root/.ssh/dbox -o StrictHostKeyChecking=no root@10.29.6.6"
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" | tee -a $W/watch181.log; }
for spec in "$@"; do
  IFS=: read mix tag ck <<< "$spec"
  [ -f eval/$mix/$tag/result.json ] && { log "$mix $tag done"; continue; }
  until $B "test -f /data/q27/w154/runs/$ck/.done"; do sleep 120; done
  if [ ! -f eval/$mix/$tag/mix.rad ]; then
    log "$mix $tag: build on box2"
    $B "cd /data/q27/w154 && source env.sh && nice -n 10 \$PY \$CODE/v3/qad181/eval181.py $tag vl/$mix eval/$mix --host /data/q27/multi/host.json --served served/$mix.json --scales runs/$ck >> eval/$mix/$tag.out 2>&1 && rm -rf eval/$mix/$tag/level" \
      || { log "$mix $tag: build FAILED"; continue; }
    mkdir -p eval/$mix/$tag
    rsync -a -e "ssh -i /root/.ssh/dbox -o StrictHostKeyChecking=no" root@10.29.6.6:/data/q27/w154/eval/$mix/$tag/{mix.rad,final.recipe,census.txt,mix.txt} eval/$mix/$tag/ \
      || { log "$mix $tag: pull FAILED"; continue; }
    mkdir -p runs/$ck && rsync -a -e "ssh -i /root/.ssh/dbox -o StrictHostKeyChecking=no" root@10.29.6.6:/data/q27/w154/runs/$ck/ runs/$ck/
  fi
  while :; do
    for c in ${CARDS//,/ }; do
      if flock -n $Q/locks/card$c.lock true; then
        bash $CODE/v3/qad181/evals181.sh $c $mix:$tag:$ck && break 2
      fi
    done
    sleep 60
  done
  log "$mix $tag scored: $(tail -n 1 $W/evals181.log)"
done
log "WATCH DONE $*"
