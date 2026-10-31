#!/bin/bash
for b in 1.40 1.60 1.80 2.00 2.20 3.00 4.00 5.00 6.00; do
  for L in 262144 196608 131072; do
    out=$(/root/fit.sh $b $L); echo "$out" | head -1
    echo "$out" | grep -q STARTS && { echo "$b $L" >> /root/caps.txt; break; }
  done
done
echo FITDONE
