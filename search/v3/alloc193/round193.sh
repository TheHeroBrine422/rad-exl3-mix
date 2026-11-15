#!/bin/bash
# round193.sh N -- RULES-193 amendment 11: one automatic round (VM; rankreg's generator + xgboost on colo via the ssh
# wrapper, nice, single-threaded). Regenerates anom / anomall / xplant / xavg / ens / rankreg from the current incumbents.
N=$1; cd /workspace/rad-exl3-mix; A=search/v3/alloc193
W=/workspace/rad-local/workers/w172; R=/workspace/rad-local/workers/lead-srch/run; C=$(git log --format=%h -1 -- $A)
G=$W/gen/r$N; mkdir -p $G; S=$W/snap3
cp $R/drv/state.jsonl $S/state.jsonl.new && mv $S/state.jsonl.new $S/state.jsonl; cp -n $R/drv/assign/*.json $S/assign/
# rankreg on colo (background): fresh state + new assignments, 24 candidates per size, top 3
( tar -C $S -cf - state.jsonl $(cd $S && find assign -newer $W/snap2/state.jsonl -name '*.json') | local/bin/ssh-r640.sh "cd /data/work/w13/w172/snap3 && tar xf - 2>/dev/null; cd /data/work/w13/w172 && sed 's/rr2-/rr$N-/g' rrgen2.sh > rrgen$N.sh && sed 's/rr2-/rr$N-/g; s/--top 4/--top 3/' rrscore2.sh > rrscore$N.sh && { ls rr$N-6/cands.jsonl >/dev/null 2>&1 || bash rrgen$N.sh; }; sleep 5; while pgrep -f '[r]ankreg193.py gen' >/dev/null; do sleep 10; done; bash rrscore$N.sh $C >/dev/null 2>&1; tar cf - rr$N-*/out/*.json" | (mkdir -p $G/rr && tar -C $G/rr -xf - --strip-components=2 2>/dev/null) ) &
nice python3 -I $A/anomtest193.py $S $W/snap1 data/alloc193/anom-snap1.json data/alloc193/pairs-snap1.jsonl $G/anom --nflags 12 --every --commit $C > $G/anom.log 2>&1
nice python3 -I $A/anomtest193.py $S $W/snap1 data/alloc193/anom-snap1.json data/alloc193/pairs-snap1.jsonl $G/anomall --nflags 12 --all --commit $C > $G/anomall.log 2>&1
nice python3 -I $A/own193.py $G/own --allocs xplant,xavg --commit $C > $G/own.log 2>&1
for w in 0.5 0.75; do for Rr in 0.25 0.5; do nice python3 -I $A/fitopt193.py $S $W/snap1 $G/ens --alloc ens2 --sizes 2,2.5,3,3.5,4,4.5,5,6 --R $Rr --w $w --tag=-w$w-R$Rr --curve data/alloc193/curve-fit-snap1.json --curv $W/curv-snap1.json --scales data/alloc193/ens-scales.json --commit $C; done; done > $G/ens.log 2>&1
wait
python3 -I $A/queue193.py $N $G/anom $G/anomall $G/own $G/rr $G/ens
