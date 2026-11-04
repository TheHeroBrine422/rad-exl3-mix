#!/bin/bash
# memguard.sh MIN_MB CMD... -- run CMD (in its own process group); while it runs, kill the whole group if the host's
# MemAvailable drops below MIN_MB. Protects a running GPU job (e.g. the 27B collector's pinned memory) from a build or
# download beside it. Exit status: CMD's, or 99 if the guard killed it.
MIN=$1; shift
setsid "$@" &
PID=$!
while kill -0 $PID 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt "$MIN" ]; then
    echo "$(date -u +%H:%M:%S) memguard: MemAvailable ${a} MB < ${MIN} MB, killing group $PID"
    kill -TERM -- -$PID; sleep 2; kill -KILL -- -$PID 2>/dev/null; wait $PID; exit 99
  fi
  sleep 2
done
wait $PID
