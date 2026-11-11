#!/bin/bash
# hold111.sh STEP... -- w111: run kl111.sh's STEPs inside ONE hold of gpubox's /root/card0.lock, /root/conv.lock and
# /root/ram.lock (exclusive: the run packs, radmixes and scores), the steps' own locks redirected to dummy files
# (KL111_LOCKDIR). Lock order card, conv, ram. ~40 min for seed 1 + 16 scores.
D=/data/w111/locks; mkdir -p $D
exec 5>/root/card0.lock; flock 5; exec 9>/root/conv.lock; flock 9; exec 7>/root/ram.lock; flock 7
echo "$(date -u +%H:%M:%S) hold111: card0 + conv + ram held: $*" >> /data/w111/result.txt
KL111_LOCKDIR=$D bash /data/w111/code/v3/e3/settle/w111/kl111.sh "$@"
rc=$?; echo "$(date -u +%H:%M:%S) hold111: released (rc $rc)" >> /data/w111/result.txt; exit $rc
