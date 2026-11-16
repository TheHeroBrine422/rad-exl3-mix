#!/bin/bash
# s196.sh JOBS (w182, #196) -- the served-path queue on box2 (eval196.py: scales applied to the virtual level, packed,
# the norms patched into a base copy, census, radiance scores on both judge sets), one job at a time, on card 0 (flock
# blocking: the search's eval host shares card 0). JOBS: lines "TAG|VL|TEMPLATE|CKPT|DEP" (CKPT empty = the untrained
# control of that level; DEP = a file that must exist first). Done when served/TAG/result.json exists. Restartable.
JOBS=$1
cd /data/q27/w182 && source env.sh
HOSTC=/data/q27/srch183/host.json
G1=/data/q27/w154/slices/golden-x181g; G2=/data/q27/w170/slices/sub-fe38
while IFS='|' read -r TAG VL TPL CKPT DEP; do
  [ -z "$TAG" ] && continue
  case "$TAG" in \#*) continue;; esac
  if [ -e served/$TAG/result.json ]; then echo "skip $TAG (done)"; continue; fi
  while [ -n "$DEP" ] && [ ! -e "$DEP" ]; do sleep 60; done
  C=""; [ -n "$CKPT" ] && C="--ckpt $CKPT"
  $PY $CODE/v3/qad196/eval196.py $TAG served --vlevel $VL --template $TPL --host $HOSTC --pack-home /data/q27/rad10-e157b/home \
     --card 0 $C --slice x181g=$G1 --slice fe38=$G2 > logs/served-$TAG.out 2>&1
  echo "$(date -u +%H:%M) served $TAG rc $?"
done < "$JOBS"
