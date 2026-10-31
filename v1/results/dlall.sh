#!/bin/bash
# every turboderp export the final ladder uses, one at a time, into /workspace/references/td-<branch>
for br in 2.00bpw 2.50bpw 3.00bpw 3.50bpw 4.00bpw 5.00bpw 6.00bpw SC_1.40bpw_H3 SC_1.60bpw_H3 SC_1.80bpw_H3 SC_2.20bpw_H3; do
  /root/dl.sh turboderp/Qwen3.8-27B-exl3 $br /workspace/references/td-$br > /root/dl-$br.out 2>&1
  echo "$br $(grep -c ^FAIL /root/dl-$br.out) fails $(du -sh /workspace/references/td-$br | cut -f1)" >> /root/dlall.out
done
echo DLALLDONE >> /root/dlall.out
