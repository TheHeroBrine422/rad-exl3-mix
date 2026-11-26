#!/bin/bash
# run232.sh (box2: cd /data/q27/w232; setsid nohup code/v3/qad232/run232.sh < /dev/null > /dev/null 2>&1 &) -- w232 part A: the xg-fin recording chain (chain232.py;
# it takes the card locks itself: lane A cards 0,1 held for the run, lane B cards 2,3 opportunistic). While it runs it writes only-card0 / only-card1 = NONE in the
# queue root (the dispatcher must not race the lane's lock polling) and removes them when it exits. Restart after a crash / reboot: the same command.
W=/data/q27/w232; Q=/data/q27/queue; cd $W; mkdir -p logs
trap 'rm -f $Q/only-card0 $Q/only-card1' EXIT
echo NONE > $Q/only-card0; echo NONE > $Q/only-card1
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') run232 start" >> $W/chain.log
TZ=America/Chicago /data/q27/venv/bin/python -I $W/code/v3/qad232/chain232.py >> $W/logs/chain.out 2>&1
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') run232: chain exited rc $?" >> $W/chain.log
