#!/bin/bash
# run79s.sh -- the 4070 queue that is left of w90-a4heal after the user's 11-08 1:10 PM CST change (#89-07 dropped,
# #89-08 deferred): wait for #89-06c's own queue (W79 HEAL DONE), then w79s.sh (the student-path reading its step 0
# calls for). Idempotent; stop: touch /data/yb/w90/STOP.
O=/data/yb/w90; mkdir -p $O
until grep -q "W79 HEAL DONE" /data/yb/w79/result.txt 2>/dev/null; do [ -f $O/STOP ] && exit 0; sleep 60; done
grep -q "W79S DONE" $O/result.txt 2>/dev/null || bash /workspace/rad-exl3-mix/proxy2b/exp/w90-a4heal/w79s.sh
