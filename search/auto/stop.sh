#!/bin/bash
# stop.sh -- stop the overnight pipeline. Kept: every scored eval, every finished quantized layer (ourq resumes per
# layer), all fits and mixes. Lost: the eval in progress, the quantization layer in progress. Resume: resume.sh.
: ${RXM_WORK:=/root} ${RXM_RADIANCE:=$RXM_WORK/install-card} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast} ${RXM_OURS:=/data/ours} ${RXM_MIX:=/data/mix}; export RXM_WORK RXM_RADIANCE RXM_KLDBIN RXM_OURS RXM_MIX   # paths (README: Paths)
cd ${RXM_WORK}; A=${RXM_WORK}/auto
for p in driver w0 w1 quant; do [ -f $A/state/$p.pid ] && kill $(cat $A/state/$p.pid) 2>/dev/null; done
sleep 2
for pat in "^${RXM_KLDBIN}" "^python -I ourq/ourq.py" "^./hproxy/hproxy" "^rad-convert" "^${RXM_RADIANCE}/bin/rad-convert"; do
  for p in $(pgrep -f "$pat"); do kill $p 2>/dev/null; done; done
sleep 5
for p in $(pgrep -f "^${RXM_KLDBIN}"); do kill -9 $p 2>/dev/null; done
# un-claim what was not finished, so a resume redoes it
for d in evq/claim/*; do t=$(basename $d); [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || rm -rf $d; done
for d in ourq/claim/K*; do [ -d $d ] || continue; K=${d##*/K}; [ -f ${RXM_OURS}/ours-K$K.rad ] || rm -rf $d; done
rm -rf ${RXM_MIX}/*.rad evq/run-*
echo "$(date +%H:%M) [auto] STOPPED (stop.sh)" | tee -a $A/auto.log >> ourchain.log
./auto/summary.sh
echo "stopped; summary in $A/SUMMARY.md"
