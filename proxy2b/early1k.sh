#!/bin/bash
# early1k.sh -- wait for the running collection's 1K snapshot, stop it and its chain, restart the chain (which now scores
# the 1K arms first, then resumes the collection from snap1024). Log /data/logs/early1k.log.
O=/data/yb
until [ -f $O/snap1024/info.json ]; do sleep 10; done
sleep 5
pgrep -f "[y]aqaBrun.sh" | xargs -r kill; sleep 1
pgrep -f "[y]aqaB.py /data/models" | xargs -r kill; sleep 5
echo "$(date -u +%T) stopped at: $(grep '^seq' $O/collect1.log | tail -1)"
mv $O/collect1.log $O/collect0.log
(setsid nohup /workspace/rad-exl3-mix/proxy2b/yaqaBrun.sh < /dev/null >> /data/logs/yaqaBrun.log 2>&1 &)
echo "$(date -u +%T) chain restarted"
