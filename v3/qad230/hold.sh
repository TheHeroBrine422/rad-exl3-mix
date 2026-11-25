#!/bin/bash
# w230: hold the box2 card locks (order 0..3 as given) until /data/q27/teach230/release exists, then release (the chain's EXIT writes it)
W=/data/q27/teach230; L=$W/hold.log
for c in "$@"; do eval "exec $((10+c))>/data/q27/locks/card$c.lock"; flock $((10+c)); echo "$(TZ=America/Chicago date "+%m-%d %-I:%M %p CST") pid $$ holds card$c" >> $L; done
while [ ! -e $W/release ]; do sleep 20; done
echo "$(TZ=America/Chicago date "+%m-%d %-I:%M %p CST") pid $$ released $@" >> $L; rm -f $W/release
