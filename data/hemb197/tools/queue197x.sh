#!/bin/bash
# queue197x.sh -- #197 (w185): the second queue, sequential like queue197.sh, through score197x.sh (EXTRA flags + VRAM
# sampler). Each item is NAME:MANIFEST:EXTRA (EXTRA may be empty). Run after queue197.sh has finished (box2 RAM: one open
# vload read at a time).
set -u
W=/data/q27/w185; G=/data/q27/w131/slices/golden-t183
export HOMEX=/data/q27/rad10-e157b/home
for item in "$@"; do
  IFS=: read -r n m x <<< "$item"
  EXTRA="$x" CARDS="0 2" $W/src/score197x.sh $n $W/manifests/$m.mix $G
done
echo "$(date -u +%FT%TZ) queue197x done: $*" >> $W/logs/score.status
