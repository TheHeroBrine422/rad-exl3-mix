#!/bin/bash
# queue2.sh -- the IQR-V1-style formats (asymmetric 5/6-bit, Lloyd 3/4/5-bit, all rotated) on 2.00 and SC 4.00.
# GPU work starts only after queue.sh AND after.sh are done (AFTERDONE); variants are prepared on the CPU meanwhile.
cd /root
F2="u6g128h u5g128h l5g128h l4g128h l3g128h"
( while ! grep -q "^4.00sc i2g128sh" /root/emb/prep.log; do sleep 30; done   # after the first 4.00sc prep: the disk, the CPU
  nice /root/embprep.sh /root/models/q38-27b-exl3-2.00bpw.rad 2.00 $F2
  nice /root/embprep.sh /root/models/q38-27b-exl3-4.00scbpw.rad 4.00sc $F2 ) &
while ! grep -q AFTERDONE /root/results.jsonl 2>/dev/null; do sleep 30; done
source <(sed -n "/^rec() {/,/^}/p; /^sweep() {/,/^}/p" /root/queue.sh)
R=/root/results.jsonl
sweep 2.00 $F2
sweep 4.00sc $F2
wait
echo Q2DONE >> $R
