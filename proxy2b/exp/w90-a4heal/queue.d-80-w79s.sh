# w90: #89-06c's student-path reading (proxy2b/exp/w90-a4heal/w79s.sh; README.md). Waits for 70-w79heal's "W79 HEAL DONE".
# Stop: touch /data/yb/w90/STOP + /data/queue.d/80-w79s.off.
CMD="bash /workspace/rad-exl3-mix/proxy2b/exp/w90-a4heal/run79s.sh"
DONE='grep -q "W79S DONE" /data/yb/w90/result.txt'
LOG=/data/yb/w90/run79s.out
MATCH="[w]90-a4heal/run79s.sh"
