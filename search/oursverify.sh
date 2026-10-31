#!/bin/bash
# oursverify.sh -- each full ours-K container, once built and proxy-scored, queued whole for calib-B verification
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}
for K in 8 7 6 5 4 3 2; do
  until grep -q -E "K$K (container \+ proxy done|quantize FAIL|conv FAIL|hproxy FAIL)|K$K: .*stopping|OURCHAINDONE" ourchain.log; do sleep 120; done
  grep -q "K$K container + proxy done" ourchain.log && echo "v-ours-K$K ${RXM_OURS}/ours-K$K.rad -" >> evq/queue-verify.txt
done
