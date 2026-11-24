#!/bin/bash
# chain228.sh (box2; setsid nohup, log /data/q27/onpol/w228/chain.log) -- w228 after the trace batch (lead 11-24): when the trace generator
# (gen.pid, target 1.6M) has exited, the calib batch on the same servers: chat cuts (render-calib-chat, target 512,000 = 32 % of 1.6M, calib's
# chat share of gen rows) and text cuts (render-calib-text, target 1,088,000) as two concurrent generators (5 + 6 slots a server); then the
# bf16 goldens (gold228.sh onpol-b1, then onpol-calib-b1). Restartable: the generators skip finished cuts, gold228.sh its finished steps.
set -u
W=/data/q27/onpol/w228; C=$W/code/v3/qad228; L=$W/chain.log; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
log "chain228 start (pid $$)"
while kill -0 $(cat $W/gen.pid) 2>/dev/null; do sleep 60; done
log "trace generator ended: $(tail -n 1 $W/out/gen.log)"
cd $W
if [ ! -e $W/calib-gen.done ]; then
  TZ=America/Chicago $PY -I $C/gen228.py render-calib-chat out-calib-chat --slots 5 --target 512000 > logs/gen-chat.out 2>&1 & p1=$!; echo $p1 > gen-chat.pid
  TZ=America/Chicago $PY -I $C/gen228.py render-calib-text out-calib-text --slots 6 --target 1088000 > logs/gen-text.out 2>&1 & p2=$!; echo $p2 > gen-text.pid
  wait $p1; r1=$?; wait $p2; r2=$?
  log "calib generators ended (rc $r1 / $r2): chat $(tail -n 1 out-calib-chat/gen.log) | text $(tail -n 1 out-calib-text/gen.log)"
  [ $r1 = 0 ] && [ $r2 = 0 ] || { log "a calib generator failed: stop"; exit 1; }
  touch $W/calib-gen.done
fi
for n in onpol-b1 onpol-calib-b1; do $C/gold228.sh $n; log "gold228 $n rc $?"; done
log "chain228 done"
