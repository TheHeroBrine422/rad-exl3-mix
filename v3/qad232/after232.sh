#!/bin/bash
# after232.sh PID (box2: cd /data/q27/w232; setsid nohup code/v3/qad232/after232.sh <pid of the running chain232.py> < /dev/null > /dev/null 2>&1 &) -- w232 hand-off after the lead's arbitration 11-26
# (w233 needs cards 2,3: lane B must not retake them): waits for the old chain (which has lane B code, its c3 / c4 claims were pre-made so it ends early) to exit, then, if it did not finish
# (no chain.done), restarts run232.sh (lane B off by default) to record the remaining chunks on cards 0,1, then runs run232x.sh (golden-v31x). Restartable steps.
W=/data/q27/w232; Q=/data/q27/queue; OLD=${1:?pid}
while kill -0 $OLD 2>/dev/null; do sleep 1; done
echo NONE > $Q/only-card0; echo NONE > $Q/only-card1
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') after232: old chain (pid $OLD) exited; chain.done $([ -e $W/chain.done ] && echo yes || echo no)" >> $W/chain.log
if [ ! -e $W/chain.done ]; then cd $W; $W/code/v3/qad232/run232.sh; fi
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') after232: xg-fin chain finished (chain.done $([ -e $W/chain.done ] && echo yes || echo no)); starting golden-v31x" >> $W/chain.log
cd /data/q27/w232x && $W/code/v3/qad232/run232x.sh
