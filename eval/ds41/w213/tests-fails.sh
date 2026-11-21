#!/bin/bash
# w213: the four in-tree fast failures again, verbose, card0 only
W=/data/q27/w213; L=$W/logs
exec {fd}> /data/q27/locks/card0.lock; flock $fd
export HIP_VISIBLE_DEVICES=0
cd $W/src/build
for t in convert_test images_test videos_test device_test; do ctest -R "^$t\$" --output-on-failure --timeout 900 > $L/t3-$t.log 2>&1; echo "$t RC=$?"; done
echo T3_DONE
