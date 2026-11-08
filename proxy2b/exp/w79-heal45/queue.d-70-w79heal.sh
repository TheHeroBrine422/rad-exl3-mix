# w79 #89-06c (healing at K4 / K5, 0.8B; proxy2b/exp/w79-heal45/README.md). NOT INSTALLED: the 4070 is off limits
# (11-08 1:20 AM CST). Install when allowed: cp to /data/queue.d/70-w79heal.sh. Stop: touch /data/yb/w79/STOP.
CMD="bash /workspace/rad-exl3-mix/proxy2b/exp/w79-heal45/run.sh"
DONE='grep -q "W79 HEAL DONE" /data/yb/w79/result.txt'
LOG=/data/yb/w79/run.out
MATCH="[w]79-heal45/run.sh"
