#!/bin/bash
# vpar.sh LIST -- parallel single-thread viterbi benches, one per logical cpu in LIST (e.g. 24-35)
cd /data/q27/w211; export E3_V2=3 E3_TB=1 OMP_NUM_THREADS=1
rm -f /tmp/vb.*.log
for c in $(echo $1 | tr ',' ' '); do
  taskset -c $c /data/q27/venv/bin/python -I vbench.py > /tmp/vb.$c.log 2>&1 &
done; wait
cat /tmp/vb.*.log | awk '{s[$1]+=$2; n[$1]++} END {for (k in s) printf "K%s %.1f ms/tile (n=%d)  ", k, s[k]/n[k], n[k]; print ""}'
