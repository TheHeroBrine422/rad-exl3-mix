#!/bin/bash
# headdry.sh -- dry run of headremake.sh's settings step outside the ladder (card 0, exclusive host RAM), so the slot
# starts on donors with a tested script. Log /data/yaqa/headfinal/run.log
: ${RXM_WORK:=/root}
exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock 7
CARD=0 STOP_AFTER=settings /data/yaqa/headremake.sh
