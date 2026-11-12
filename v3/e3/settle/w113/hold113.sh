#!/bin/bash
# hold113.sh UNIT... -- w113: run a113.sh's UNITs inside ONE hold of gpubox's /root/card0.lock then /root/ram.lock (shared),
# the units' own locks redirected to dummy files (A113_LOCKDIR). For short batches (~4 units, ~5 min) so the card-0
# queue (w110's packs) is not held for long.
D=/data/w113/locks; mkdir -p $D
exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
echo "$(date -u +%H:%M:%S) hold113: card0 + ram (shared) held: $*" >> /data/w113/a/result.txt
A113_LOCKDIR=$D bash /data/w113/code/v3/e3/settle/w113/a113.sh "$@"
rc=$?; echo "$(date -u +%H:%M:%S) hold113: released (rc $rc)" >> /data/w113/a/result.txt; exit $rc
