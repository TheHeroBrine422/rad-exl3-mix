#!/bin/bash
# queue3.sh -- e2e of the real Lloyd-3 SC 2.00, then the real Lloyd 3..6 sweep on SC 4.00 (each built by --reuse in ~1 min)
cd /root
/root/e2e.sh /root/models/final/q38-27b-sc2.00-e3.rad sc2.00-e3
for B in 4 5 6 3; do
  /root/convemb.sh sc4.00-e$B /workspace/references/td-SC_4.00bpw_H5 $B /root/models/q38-27b-exl3-4.00scbpw.rad
  [ -f /root/models/final/q38-27b-sc4.00-e$B.rad ] && /root/e2e.sh /root/models/final/q38-27b-sc4.00-e$B.rad sc4.00-e$B
done
echo Q3DONE >> /root/e2e.jsonl
