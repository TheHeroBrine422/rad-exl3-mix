#!/bin/bash
# startfinal.sh -- start the trunk requant (user's go only): both cards on queue-final.txt. To take a card out (user's
# call, nothing scheduled): touch /data/v3ladder2/stop-c<CARD>; it leaves after its current level. Log /data/v3ladder2/
: ${RXM_WORK:=/root}
D=/data/v3ladder2
mkdir -p $D; [ -z "$(ls -d $D/K* $D/mul1/K* 2>/dev/null)" ] || { echo "$D has levels already: refusing"; exit 1; }
rm -f $D/stop-c*; cp $RXM_WORK/ourq/queue-final.txt $D/queue.txt
cd $RXM_WORK/ourq
(QUEUE=$D/queue.txt setsid nohup ./ladder2.sh 0 > $D/driver-c0.out 2>&1 < /dev/null &)
sleep 90                                              # card 1 starts after card 0 has popped the first level
(QUEUE=$D/queue.txt setsid nohup ./ladder2.sh 1 > $D/driver-c1.out 2>&1 < /dev/null &)
echo "started $(date -u +%T) UTC (both cards; stop one with: touch $D/stop-c<CARD>)"
