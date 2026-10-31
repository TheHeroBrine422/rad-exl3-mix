#!/bin/bash
# wait for the 25 probes, then fit 3 model variants (fit only, 5 folds) in parallel
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}
until [ $(for t in $(awk '{print $1}' auto/q.prio | grep 'w1$'); do [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo x; done | wc -l) = 0 ]; do sleep 30; done
cp evq/results.jsonl ${RXM_WORK}/nf/kt/results-snap.jsonl
cd knap
for v in base:0:8 edge:1:8 edge6:1:6; do IFS=: read n e k <<< "$v"
  ( OMP_NUM_THREADS=4 KFIT_EDGE=$e KFIT_KNOT=$k KFIT_OURS_ONLY=1 KFIT_BOOT=0 KFIT_FOLDS=5 KFIT_VERBOSE=1 KFIT_TAG=ec$n- \
    nice python3 -W ignore ${RXM_WORK}/nf/kt/kfit2-edge.py ${RXM_WORK}/knap > ${RXM_WORK}/nf/kt/ec-$n.out 2> ${RXM_WORK}/nf/kt/ec-$n.txt ) &
done
wait; echo EDGECMPDONE > ${RXM_WORK}/nf/kt/ec.done
