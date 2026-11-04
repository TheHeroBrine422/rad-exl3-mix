#!/bin/bash
# after116.sh -- once the radiance 1.16.0 build is done, fetch the calib input Hessians (missed by the rest step) under memguard.
while pgrep -x memguard.sh > /dev/null || pgrep -f "^/bin/bash ./build.sh /data/rh116" > /dev/null; do sleep 60; done
echo "$(date -u +%H:%M:%S) build done, hessians download"
cd /data/ourq && ./memguard.sh 3000 nice -n 19 env HF_XET_HIGH_PERFORMANCE=0 /data/venv-exl3/bin/hf download StillDeadcode/qwen3.8-27b-quant-challenge \
  --repo-type dataset --revision 2dcf0061cb683152742b904e2965ce7cd92b2da6 --local-dir /data/challenge --include "hessians/calib/*" --max-workers 2
echo "$(date -u +%H:%M:%S) download exit $?"; du -sh /data/challenge/hessians
