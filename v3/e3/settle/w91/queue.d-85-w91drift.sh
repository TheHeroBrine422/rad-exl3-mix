# w91 item 2 (0.8B drift / variance split + bias oracle; v3/e3/settle/w91.md). Runs the snapshot /data/yb/w91/snap/q91.sh
# (idempotent; re-snapshot after editing the repo copy). Stop: touch /data/yb/w91/STOP + /data/queue.d/85-w91drift.off.
CMD="bash /data/yb/w91/snap/q91.sh"
DONE='grep -q "Q91 DONE" /data/yb/w91/result.txt'
LOG=/data/yb/w91/run.out
MATCH="[w]91/snap/q91.sh"
