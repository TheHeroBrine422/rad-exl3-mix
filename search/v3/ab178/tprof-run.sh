#!/bin/bash
# tprof-run.sh SIZE -- RULES-178 timing on ds8 (w142): arm B's container at SIZE (own copy), tprof.sh on a queued card
# (card lock + ram.lock shared), then tprof.py. The container is deleted afterwards.
set -u
Q=/data/q27; W=$Q/w142; s=$1
declare -A TAG=([2.5]=v3s171-2.5-block-00626 [4.0]=v3s171-4-block-00420)
t=${TAG[$s]}; mf=$W/mix/tprof-$s.mix; out=$W/mix/tprof-$s.rad
{ cat $Q/mixes/$t.mix; awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $Q/mixes/$t.mix; } > $mf
[ -f $out ] || { flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $Q/tools/radmix $out.part $Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad $mf > /dev/null && mv $out.part $out; }
c=${2:-3}
flock $Q/locks/card$c.lock flock -s $Q/locks/ram.lock bash $W/code/tprof.sh $out $c $W/tprof-$s.log $W/prompt8k.txt
/data/q27/venv/bin/python $W/code/tprof.py $W/tprof-$s.log > $W/tprof-$s.out 2>&1
rm -f $out
