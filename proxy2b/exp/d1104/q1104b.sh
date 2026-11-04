#!/bin/bash
# q1104b.sh -- 4070 queue (2026-11-04 morning): rerun the heal0 K2 seed-0 arm lost to the full disk, then hconv (#33-25)
# on the synced snapshots. Both idempotent; stop with touch /data/yb/night.stop.
R=/workspace/rad-exl3-mix
rm -rf /data/yb/d1104/h-k2edyswjasym_r0 /data/yb/d1104/h-k2edyswjasym_r0.log
D1104_STEPS=heal0 bash $R/proxy2b/exp/d1104/run.sh
[ -f /data/yb/night.stop ] && exit 0
bash $R/v3/yaqa/hconv/hconv.sh
