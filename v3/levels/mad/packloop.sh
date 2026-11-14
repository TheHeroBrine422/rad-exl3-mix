#!/bin/bash
# packloop.sh HOME -- (ds8) packall-mad.sh HOME 1 over every mad level, menu-critical first (K2-5, 5.25-7.5, 1-1.875, sub-1),
# every 10 min until all 49 packs exist (levels become packable as their swj / DONE lands); one rad-convert at a time at
# nice 19 beside the CPU-bound quant lanes. Stop: touch /data/q27/mad/STOP. Process name `bash /data/q27/mad/tools/packloop.sh`.
# (#177-03, w151; pack home = radiance qc-exl3-v3-mad e054e71 plugins on rad10-perf, identity-checked against w150's K2.5)
Q=/data/q27; T=$Q/mad/tools
ORD="2 2.125 2.25 2.375 2.5 2.625 2.75 2.875 3 3.125 3.25 3.375 3.5 3.625 3.75 3.875 4 4.125 4.25 4.375 4.5 4.625 4.75 4.875 5
5.25 5.5 5.75 6 6.25 6.5 6.75 7 7.5 1.875 1.75 1.625 1.5 1.375 1.25 1.125 1 0.875 0.75 0.625 0.5 0.375 0.25 0.125"
while [ ! -f $Q/mad/STOP ]; do
  nice -n 19 bash $T/packall-mad.sh $1 1 $ORD
  [ $(ls $Q/packs-mad/*.rad 2>/dev/null | wc -l) -ge 49 ] && { echo "$(date -u +%T) PACKLOOP: 49 packs" >> $Q/mad/logs/packall.log; exit 0; }
  sleep 600
done
