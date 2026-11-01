#!/bin/bash
# headhalf.sh -- fractional head donors with NOTHING else running (user, 2026-11-01). Log /data/yaqa/headslab2/half.log
#  waits for cardzero.sh to finish, stops both ladders, then K3.5 alone (peak host RSS logged); if it works, K4.5 5.5
#  6.5 7.5, then the slab-head curve re-planned / re-priced with all half-step donors (two headsim passes: VRAM).
#  Restarts both ladders at the end (also on failure).
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M) $*" | tee -a $H/half.log; }
LP='^/data/venv-exl3/bin/python -I ourq/ourq.py -K [0-9.]* -o /data/v3levels/K'
restart() {
  (setsid nohup /data/v3levels/levels.sh 0 "3.25 3.75 2.5 4.25 2.25 4.75 5 6" > /dev/null 2>&1 < /dev/null 9>&- &)
  (setsid nohup /data/v3levels/levels.sh 1 "2.75 3.5 3 4 4.5 2 5.5" > /dev/null 2>&1 < /dev/null 9>&- &)
  log "both ladders restarted"
}
until grep -q "card0 done" $H/card0.log 2>/dev/null || ! pgrep -f "^/bin/bash ./cardzero.sh" > /dev/null; do sleep 15; done
pkill -f "^/bin/bash /data/v3levels/levels.sh "; sleep 1; pkill -f "$LP"
while pgrep -f "$LP" > /dev/null; do sleep 2; done; sleep 5
trap restart EXIT
log "everything else stopped: $(free -m | awk 'NR==2{print $7}') MB available"
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
hq() {
  [ -f $H/K$1/wq/lm_head.rowerr.npz ] && return 0
  rm -rf $H/K$1
  $PY -I ourq/ourq.py -K $1 -o $H/K$1 --only-head --device 0 --save-wq --head-perm $H/perm.npy --frac-ext /data/exl3frac/src \
    > $H/K$1.log 2>&1 &
  local p=$! peak=0
  while kill -0 $p 2>/dev/null; do
    r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$r" ] && [ $r -gt $peak ] && peak=$r; sleep 2; done
  wait $p; local rc=$?
  [ -f $H/K$1/wq/lm_head.rowerr.npz ] && { log "K$1 head OK, peak RSS $((peak / 1024)) MB"; return 0; }
  log "K$1 head FAIL rc=$rc, peak RSS $((peak / 1024)) MB, oom_kill $(grep oom_kill /sys/fs/cgroup/memory.events | cut -d' ' -f2)"; return 1
}
hq 3.5 || { log "K3.5 fails even alone: stop (integer curve stands)"; exit 1; }
for K in 4.5 5.5 6.5 7.5; do hq $K; done
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
export HS_TRAIN=qad HS_FLOOR=3
HS_KS=3,3.5,4,4.5,5,5.5,6 HS_BUDGETS=$(bud 3.5 5.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-A-plan.json >> $H/half.log 2>&1
HS_KS=5,5.5,6,6.5,7,7.5,8 HS_BUDGETS=$(bud 5.5 7.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-B-plan.json >> $H/half.log 2>&1
for p in A B; do
  $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/curve-$p-plan.json $CK $H/curve-$p.json > $H/curve-$p.json.log 2>&1
  [ -s $H/curve-$p.json ] && log "half-step curve $p priced" || log "curve $p headsim FAIL"
done
log "headhalf done"
