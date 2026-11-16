#!/bin/bash
# queue197.sh -- #197 (w185): the arms, one read at a time (box2 RAM: one vload open holds the planes), each on the first
# free of cards 0 / 2 (cards 1 / 3 are w182's QAD). Restartable: score197.sh skips finished reads (rows present).
set -u
W=/data/q27/w185; G=/data/q27/w131/slices/golden-t183
export HOMEX=${HOMEX:-/data/q27/rad10-e157b/home}   # the search engine's plugin home (the vload build's own home has no plugins)
for n in "$@"; do
  CARDS="0 2" $W/src/score197.sh $n $W/manifests/$n.mix $G
done
echo "$(date -u +%FT%TZ) queue done: $*" >> $W/logs/score.status
