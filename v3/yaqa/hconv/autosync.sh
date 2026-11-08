#!/bin/bash
# autosync.sh -- (VM) every 15 min: sync new snapshots to the 4070 (sync.sh); if any arrived and hconv is not running,
# start hconv.sh there (idempotent: only new arms + rescoring). Stops after the 2,048 snapshot or on touch /tmp/hconv-autosync.stop.
B=/workspace/rad-local/bin; V=/workspace/rad-exl3-mix/v3/yaqa/hconv
while [ ! -f /tmp/hconv-autosync.stop ]; do
  out=$(timeout 600 $V/sync.sh 2>&1); echo "$(date -u +%T) $out"
  echo "$out" | grep -q synced && pending=1
  if [ -n "$pending" ] && ! $B/ssh-4070.sh 'pgrep -f "^bash /workspace/rad-exl3-mix/v3/yaqa/hconv/[h]conv.sh"' < /dev/null > /dev/null; then
    $B/ssh-4070.sh '(cd /tmp && setsid nohup bash /workspace/rad-exl3-mix/v3/yaqa/hconv/hconv.sh >> /data/yb/hconv.log 2>&1 < /dev/null &)' < /dev/null
    pending=; echo "$(date -u +%T) hconv kicked"
  fi
  echo "$out" | grep -q "hdiag-snap2048-dy" && break
  sleep 900
done
