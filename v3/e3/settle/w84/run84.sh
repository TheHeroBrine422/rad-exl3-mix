#!/bin/bash
# run84.sh -- w84's gpubox queue (#117, ../w84.md): G2 (signchk) -> pair p0 (base) -> pair p1 (s1) -> verdict.
# Restartable: every step of w84.sh skips what is done. Log /data/w84/result.txt; last line "W84 QUEUE DONE" or
# "W84 QUEUE FAIL ...". Stop: kill this script (and its w84.sh child); a rerun resumes.
A=/data/w84; W=$A/code/v3/e3/settle/w84
echo "$(date -u +%T) W84 QUEUE start" >> $A/result.txt
bash $W/w84.sh chk || { echo "$(date -u +%T) W84 QUEUE FAIL (G2: $A/chk/signchk84.log)" >> $A/result.txt; exit 1; }
for arm in p0 p1; do
  bash $W/w84.sh $arm || { echo "$(date -u +%T) W84 QUEUE FAIL ($arm: $A/$arm/w84.log)" >> $A/result.txt; exit 1; }
done
bash $W/w84.sh verdict
echo "$(date -u +%T) W84 QUEUE DONE" >> $A/result.txt
