#!/bin/bash
# donors144all.sh LIST -- (w144, #174-02, ds8) every trellis embedding donor of LIST (lines "NAME K CB": t-i8 4 mul1i8):
# the level's src pulled from box2 over the 10 GbE link when it was quantized there, then donor144.sh; each finished
# donor pushed to box2's /data/q27/emb/trellis/ too (the brief: donors on box2 + ds8). Restartable (skips finished
# donors / copies). Log W/logs/donors144.log.
set -u
Q=/data/q27 W=/data/q27/w144 D=/data/q27/emb/trellis
. $W/link144.env   # S / R: ssh (-n) and rsync transport to box2 over the 10 GbE link, B: box2 there -- machine-local, not in git
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/logs/donors144.log; }
while read -r name k cb; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  src=$D/$name/K$k/src
  if [ ! -f $src/q-embed.safetensors ]; then
    $S $B "[ -f $src/q-embed.safetensors ]" || { log "$name K$k: no src on either host yet"; continue; }
    mkdir -p $D/$name/K$k; rsync -a -e "$R" $B:$src $D/$name/K$k/ < /dev/null || { log "$name K$k: pull failed"; continue; }
  fi
  if [ ! -f $D/emb-$name-K$k.rad ]; then
    r=$(bash $W/donor144.sh $name $k $cb < /dev/null 2>&1 | tail -n 1); log "$name K$k: $r"
  fi
  [ -f $D/emb-$name-K$k.rad ] || continue
  $S $B "[ -f $D/emb-$name-K$k.rad ]" || { rsync -a --partial -e "$R" $D/emb-$name-K$k.rad $B:$D/ < /dev/null && log "$name K$k: donor pushed to box2"; }
done < $1
log "list $1 done"
