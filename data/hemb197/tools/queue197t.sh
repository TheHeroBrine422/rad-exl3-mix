#!/bin/bash
# queue197t.sh NAME:MANIFEST ... -- #197 (w185): the second-slice check (t186n, 36,457 positions, 22 documents) for the
# head arms, sequential, through score197x.sh. Run after the t183 queues (box2 cards are shared with w182 / w186).
set -u
W=/data/q27/w185; G=/data/q27/w164/slices/golden-t186n
export HOMEX=/data/q27/rad10-e157b/home
for item in "$@"; do
  IFS=: read -r n m <<< "$item"
  CARDS="0 2" $W/src/score197x.sh $n $W/manifests/$m.mix $G
done
echo "$(date -u +%FT%TZ) queue197t done: $*" >> $W/logs/score.status
