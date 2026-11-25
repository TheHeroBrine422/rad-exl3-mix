#!/bin/bash
# run230.sh (box2: cd /data/q27/teach230; setsid nohup code/v3/qad230/run230.sh < /dev/null &) -- w230: take the four card locks (hold.sh), run
# the chain (chain230.py: lanes A = cards 0,1 / B = cards 2,3, generation + TP2 bf16 recording, restartable), release the locks on exit.
# Restart after a crash / reboot: the same command (finished cuts, chunks and sets are skipped).
W=/data/q27/teach230; cd $W; mkdir -p logs
rm -f $W/release $W/hold.log
setsid nohup $W/code/v3/qad230/hold.sh 0 1 2 3 < /dev/null > /dev/null 2>&1 &
HP=$!
trap 'touch $W/release; sleep 25; kill $HP 2>/dev/null' EXIT
until [ "$(grep -c "holds card" $W/hold.log 2>/dev/null)" -ge 4 ] && grep -q "holds card3" $W/hold.log; do sleep 10; done
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') run230: 4 card locks held" >> $W/chain.log
TZ=America/Chicago /data/q27/venv/bin/python -I $W/code/v3/qad230/chain230.py >> $W/logs/chain.out 2>&1
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') run230: chain exited rc $?" >> $W/chain.log
