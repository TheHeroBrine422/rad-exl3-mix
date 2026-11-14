#!/bin/bash
# q10.sh -- card 1, after q9: exl3i's device tests of mres (exl3i version 5) on ROCm 10.1 and 7.14 (q9's ctall ran
# nothing: gt.sh had no execute bit). (w152)
W=/data/q27/w152; cd $W
while pgrep -f "^bash tools/q9[.]sh" > /dev/null; do sleep 20; done
bash tools/ctall.sh 1 10 mres; bash tools/ctall.sh 1 714 mres
echo "$(date -u +%T) q10 DONE" >> logs/q.log
