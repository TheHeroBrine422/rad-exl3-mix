#!/bin/bash
# qnext.sh -- card 1: q5, q6, q7 once q8 is done (re-ordered 11-13 10:30 PM CST: the mad decode question first). (w152)
W=/data/q27/w152; cd $W
while pgrep -f "^bash tools/q8[.]sh" > /dev/null; do sleep 20; done
bash tools/q5.sh; bash tools/q6.sh; bash tools/q7.sh
echo "$(date -u +%T) qnext DONE" >> logs/q.log
