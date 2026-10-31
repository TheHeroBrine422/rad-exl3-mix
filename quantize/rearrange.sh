#!/bin/bash
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}
for p in $(pgrep -f "^/bin/bash ./ourq/retryhead.sh"); do kill $p; done
for p in $(pgrep -f "^/bin/bash ./ourq/ourchain2.sh 0"); do kill $p; done
sleep 1
for p in $(pgrep -f "^python -I ourq/ourq.py"); do kill $p; done
sleep 5
rm -rf ourq/claim/K8
mv -f /tmp/ourchain2.new ourq/ourchain2.sh; mv -f /tmp/gate0.new ourq/gate0.sh; chmod +x ourq/ourchain2.sh ourq/gate0.sh
HEADROOM=4096 nohup ./hproxy/evalq.sh 0 evq/queue-cal.txt golden-calA corpus/calA.jsonl > evq/worker0.log 2>&1 < /dev/null &
nohup ./ourq/gate0.sh > ${RXM_WORK}/gate0.out 2>&1 < /dev/null &
echo "$(date +%H:%M) rearranged: card 0 on calibration, then frontier verification, then full models; head retries dropped" >> ourchain.log
