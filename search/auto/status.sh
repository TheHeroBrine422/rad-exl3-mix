#!/bin/bash
# status.sh -- one-screen view of the pipeline
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}; A=${RXM_WORK}/auto
date '+%H:%M card time'; echo "deadline $(date -d @$(cat $A/state/deadline) '+%m-%d %H:%M'), round $(cat $A/state/round 2>/dev/null)"
for p in driver w0 w1 quant; do printf "%-7s %s\n" $p "$([ -f $A/state/$p.pid ] && kill -0 $(cat $A/state/$p.pid) 2>/dev/null && echo running || echo -)"; done
echo "queue: $(awk '$1!="END"' $A/q | wc -l) lines, prio $(wc -l < $A/q.prio); running: $(ps -eo args | grep "^${RXM_WORK}/radiance-kldgold" | grep -o 'mix/[^ ]*' | tr '\n' ' ')"
tail -n 8 $A/auto.log
