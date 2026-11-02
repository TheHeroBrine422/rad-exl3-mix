#!/bin/bash
# startfinal.sh -- start the trunk requant (user's go only): both cards on queue-final.txt; card 1 gets a stop file
# after STOP1_H hours (default 12) and leaves after its current level, card 0 continues alone. Log /data/v3ladder2/
: ${RXM_WORK:=/root}
D=/data/v3ladder2 H=${STOP1_H:-12}
mkdir -p $D; [ -z "$(ls -d $D/K* $D/mul1/K* 2>/dev/null)" ] || { echo "$D has levels already: refusing"; exit 1; }
rm -f $D/stop-c*; cp $RXM_WORK/ourq/queue-final.txt $D/queue.txt
cd $RXM_WORK/ourq
(QUEUE=$D/queue.txt setsid nohup ./ladder2.sh 0 > $D/driver-c0.out 2>&1 < /dev/null &)
sleep 90                                              # card 1 starts after card 0 has popped the first level
(QUEUE=$D/queue.txt setsid nohup ./ladder2.sh 1 > $D/driver-c1.out 2>&1 < /dev/null &)
(setsid nohup bash -c "sleep $((H * 3600)); touch $D/stop-c1" > /dev/null 2>&1 < /dev/null &)
echo "started $(date -u +%T) UTC; card 1 stops after its level in progress at $(date -u -d "+$H hour" +%T) UTC"
