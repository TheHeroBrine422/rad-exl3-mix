#!/bin/bash
# after round 3 on calib-A: the report and a full refit (CV, bootstrap 20, Thompson 3) into $RXM_WORK/knap/r3*.txt
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}
pend() { awk "\$1!=\"END\"{print \$1}" evq/queue-r3.txt | while read t; do [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done; }
until [ -z "$(pend)" ]; do sleep 60; done
python3 -I knap/r3report.py > knap/r3report.txt 2>&1
cd knap && KFIT_TAG=r4- KFIT_BOOT=20 KFIT_THOMPSON=3 KFIT_FOLDS=10 python3 -W ignore kfit2.py ${RXM_WORK}/knap 1.6,1.8,2.0,2.25,2.5,2.75,2.99,3.25,3.5,3.75,3.99,4.5,5.0,5.49,6.0 > r3fit-frontier.txt 2> r3fit.txt
echo "$(date +%H:%M) round 3 done: knap/r3report.txt, knap/r3fit.txt, r4 candidate mixes (not queued)" >> ${RXM_WORK}/ourchain.log
