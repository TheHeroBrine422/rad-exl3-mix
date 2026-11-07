#!/bin/bash
# #104 amendment 2: swj's K4 (and K2.5) gain re-scored on a fresh calib-verify slice (w59, v3/yaqa/SWJ27.md,
# v3/yaqa/verify27/README.md). Runs as a q2.sh insert, before the next arm. Card 0 ~35 min: 12 scores (10 arms on
# golden-verify27, 2 head-matched K4 controls on golden-search27), each under card0.lock + ram.lock shared. The golden
# and the head-matched mixes were built in the dry run (no card); v27.sh redoes any missing piece. Restartable: q2
# marks this insert done whatever its exit, so after a FAIL rerun it by hand (`bash $0`); it resumes per arm.
cd /data/verify27 || exit 1
bash /data/verify27/tools/v27.sh < /dev/null
rc=$?; [ $rc = 0 ] && m="$(grep -h '^VERDICT' /data/verify27/report.txt)" || m="FAIL exit $rc (/data/verify27/result.txt)"
echo "$(date -u +%T) verify27 $m" >> /data/q1106/q.log
