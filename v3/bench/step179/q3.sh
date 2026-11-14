#!/bin/bash
# q3.sh -- after q2 on card 1: the timing diagnostic (diag-step2 home, outputs garbage): e2e with every non-GEMV op of the
# plugin's own skipped vs none skipped, both packs. (w152)
W=/data/q27/w152; cd $W
while pgrep -f "^bash tools/q[12].sh" > /dev/null; do sleep 20; done
ROWS="dg0=$W/b10diag-step2/home" bash tools/e2.sh 1 9 v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8
SRVENV="QWEN35_EXL3_DIAG_SKIP=nongemv" ROWS="dgskip=$W/b10diag-step2/home" bash tools/e2.sh 1 9 v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8
echo "$(date -u +%T) q3 DONE" >> logs/q.log
