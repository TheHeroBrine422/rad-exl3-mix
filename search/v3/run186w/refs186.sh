#!/bin/bash
# refs186.sh -- the #186 reference lane (w164): references scored on golden-t186n, one at a time (ref186.sh).
#   LIST       names to score (default: V2 1.50-7.00 and IQR-V4 iq2r-iq6r); done names (rows present) are skipped
#   LOCAL_DIRS where containers already sit (searched first; default ds8's /data/q27/mixes and /data/q27/w153/ref)
#   CARDS      cards to take, first free (default 7 on ds8; box2: CARDS="3 2")
# A container not found locally is downloaded with hf (waits for 200 GB free), scored, deleted. Restartable.
# Rerun on box2 (w164, 11-15): LIST="iq2r iq4r v2-6.00 v2-7.00 iq3r iq5r iq6r" CARDS="3 2" \
#   LOCAL_DIRS="/data/q27/w138/hf/iq2r /data/q27/w138/hf/iqr-v4 /data/q27/mixes" ./refs186.sh
cd /data/q27/w164/refs; G=${G:-/data/q27/w164/slices/golden-t186n}; D=/data/q27/w164/refdl; mkdir -p $D
V2=Hero422/qwen3.8-27b-exl3-v2-rad; IQ=StillDeadcode/qwen3.8-27b-iqr-v4
L=${LIST:-"v2-1.50 v2-2.00 v2-2.50 v2-3.00 v2-3.50 v2-4.00 v2-4.50 v2-5.00 v2-6.00 v2-7.00 iq2r iq3r iq4r iq5r iq6r"}
for n in $L; do
  [ -s $n-t186n.json.rows ] && continue
  if [[ $n == v2-* ]]; then f=qwen3.8-27b-exl3-v2-${n#v2-}bpw.rad; r=$V2; else f=qwen3.8-27b-$n-v4.rad; r=$IQ; fi
  loc=""; for d in ${LOCAL_DIRS:-/data/q27/mixes /data/q27/w153/ref}; do [ -f $d/$f ] && loc=$d/$f && break; done
  dl=0; if [ -z "$loc" ]; then
    while [ $(df --output=avail -B1G /data | tail -1) -lt 200 ]; do sleep 60; done
    hf download $r $f --local-dir $D > /dev/null 2>&1 || { echo "$n download failed"; continue; }; loc=$D/$f; dl=1; fi
  ./ref186.sh $n-t186n $loc $G; echo "$(date +%T) $n done rc $?"
  [ $dl = 1 ] && rm -f $loc
done
echo "LANE DONE"
