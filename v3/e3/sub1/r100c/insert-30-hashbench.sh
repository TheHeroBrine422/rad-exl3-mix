#!/bin/bash
# #100c hash vs mul1i8 decode microbench (w48, v3/e3/sub1/r100c.md): w29's dev builds on card 0, guarded (idle card
# only), <= 20 min hard cap. Runs as a q2.sh insert, before the next arm.
timeout -k 15 1200 bash /data/tp/w48/hashbench.sh >> /data/tp/w48/hashbench.out 2>&1 < /dev/null
echo "$(date -u +%T) hashbench exit $? ($(ls -d /data/tp/w48/res/p? 2>/dev/null | wc -l) clean passes)" >> /data/q1106/q.log
