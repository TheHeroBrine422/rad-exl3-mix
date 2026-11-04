#!/bin/bash
# q1104.sh -- 4070 queue 2026-11-04 (one GPU job at a time): w11's healfix (heal arms re-scored with their folded
# norms), w11's xseed (does #87's correction transfer across sibling seeds?), w12's sib27 GPU equivalence test.
L=/data/yb/q1104.log; log() { echo "$(date -u +%T) $*" | tee -a $L; }
while pgrep -f "quantize/[o]urq.py|proxy2b/[e]valq.py|yaqa/[c]ollect.py" > /dev/null; do sleep 60; done
for s in proxy2b/exp/w11-review/healfix.sh proxy2b/exp/w11-review/xseed.sh v3/sib27/test_gpu.sh; do
  [ -f /data/yb/night.stop ] && { log "stop file"; exit 0; }
  log "start $s"; bash /workspace/rad-exl3-mix/$s > /data/logs/q1104-$(basename $s .sh).log 2>&1; log "done $s rc $?"
done
log "Q1104 DONE"
