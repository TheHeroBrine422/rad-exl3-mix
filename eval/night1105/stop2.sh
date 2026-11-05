#!/bin/bash
# stop2.sh -- end tp2night.sh's two-card part now: no new TP2 stage starts, the collector stops after its current
# sequence (resumable), a running record chunk / server is killed (that chunk reruns later). Card 1 is free when
# night.log shows "STOP2" (or stage 4 starting); card 0 continues alone.
N=/data/night1105; touch $N/STOP2; [ -d $N/hdiag-D ] && touch $N/hdiag-D/STOP
[ -f $N/tp2.pid ] && kill $(cat $N/tp2.pid) 2>/dev/null
[ -f /data/swap27/card1-quant.pid ] && kill $(cat /data/swap27/card1-quant.pid) 2>/dev/null   # card 1's quant chunk goes back to the queue
echo "$(date -u +%T) stop2 requested" >> $N/night.log; echo "stop requested; card 1 free in ~1 min (collector: after its current sequence, <1 min)"
