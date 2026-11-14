#!/bin/bash
# runloop.sh CONF TAG [MAX] -- (ds8 / box2) rerun `rxrun.run CONF` until it exits 0, at most MAX times (default 6): a
# killed or faulted run resumes where it stopped (rxrun skips finished jobs / steps). Logs /data/q27/mad/logs/TAG.{log,loop.log}.
# Stop: touch /data/q27/mad/STOP (checked between attempts) or the conf root's HOLD file (rxrun's own). (#177-03, w151)
Q=/data/q27; C=$1 T=$2 M=${3:-6}; L=$Q/mad/logs; mkdir -p $L
for a in $(seq $M); do
  [ -f $Q/mad/STOP ] && { echo "$(date -u +%T) STOP" >> $L/$T.loop.log; exit 2; }
  echo "$(date -u +%T) attempt $a" >> $L/$T.loop.log
  env PYTHONPATH=$Q/stage/code/pipeline $Q/venv/bin/python -m rxrun.run $C >> $L/$T.log 2>&1 && { echo "$(date -u +%T) OK" >> $L/$T.loop.log; exit 0; }
  sleep 30
done
echo "$(date -u +%T) GAVE UP" >> $L/$T.loop.log; exit 1
