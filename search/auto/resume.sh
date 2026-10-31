#!/bin/bash
# resume.sh -- restart the overnight pipeline after stop.sh (or a reboot). HOURS=N extends the deadline to now+N.
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}; A=${RXM_WORK}/auto
[ -f $A/state/driver.pid ] && kill -0 $(cat $A/state/driver.pid) 2>/dev/null && { echo "already running"; exit 0; }
[ -n "$HOURS" ] && echo $(( $(date +%s) + HOURS * 3600 )) > $A/state/deadline
nohup ./auto/auto.sh > $A/driver.out 2>&1 < /dev/null &
echo "$(date +%H:%M) [auto] resumed (pid $!)" | tee -a $A/auto.log >> ourchain.log
