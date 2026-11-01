#!/bin/bash
# headalone.sh K -- one head quant with NOTHING else on the box (user, 2026-11-01), peak host RSS and OOM count logged.
# Log /data/yaqa/headslab2/alone.log; output /data/yaqa/headslab2/K<K>.
: ${RXM_WORK:=/root}; export RXM_WORK
H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python K=$1
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/alone.log; }
exec 9>/root/card0.lock; flock 9; exec 7>/root/ram.lock; flock 7
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
rm -rf $H/K$K
log "K$K start: $(free -m | awk 'NR==2{print $7}') MB available, oom_kill $(grep oom_kill /sys/fs/cgroup/memory.events | cut -d' ' -f2)"
$PY -X faulthandler -I ourq/ourq.py -K $K -o $H/K$K --only-head --device 0 --save-wq --head-perm $H/perm.npy \
  --frac-ext /data/exl3frac/src > $H/K$K.log 2>&1 &
p=$! peak=0 swp=0
while kill -0 $p 2>/dev/null; do
  r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); w=$(awk '/VmSwap/{print $2}' /proc/$p/status 2>/dev/null)
  [ -n "$r" ] && [ $r -gt $peak ] && peak=$r; [ -n "$w" ] && [ $w -gt $swp ] && swp=$w; sleep 1; done
wait $p; rc=$?
log "K$K rc=$rc peak RSS $((peak / 1024)) MB, peak swap $((swp / 1024)) MB, oom_kill $(grep oom_kill /sys/fs/cgroup/memory.events | cut -d' ' -f2), $( [ -f $H/K$K/wq/lm_head.rowerr.npz ] && echo OK || echo FAIL)"
