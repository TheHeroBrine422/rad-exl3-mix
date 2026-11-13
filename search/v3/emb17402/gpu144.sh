#!/bin/bash
# gpu144.sh [ARGS] -- (w144, #174-02, box2 / ds8) exl3dense_embed_exl3i_gpu_test (device row == host row bit for bit, then
# the gather's time at the 27B's table beside embed_tq) on the first free card (polling the card locks, as quant144.sh). Output W/logs/gpu144-<host>-<time>.txt. ARGS go to the test (--bench-only, --ids N).
set -u
Q=/data/q27 W=/data/q27/w144 LK=/data/q27/locks
if [ -f $Q/tools/env-ds8.sh ]; then H=ds8; CARDS="0 1 2 3 4 5 6 7"; else H=box2; CARDS="0 1 2 3"; fi
O=$W/logs/gpu144-$H-$(date -u +%m%d%H%M).txt
while :; do                          # the first free card by polling its lock, held on fd 9 until the end
  c=
  for cc in $CARDS; do exec 9>$LK/card$cc.lock; if flock -n 9; then c=$cc; break; fi; exec 9>&-; done
  [ -n "$c" ] && break; sleep 2
done
echo "host $H card $c $(date -u)" > $O
ROCR_VISIBLE_DEVICES=$c $W/pb/exl3dense/exl3dense_embed_exl3i_gpu_test "$@" >> $O 2>&1; echo "rc $?" >> $O
exec 9>&-
echo $O
