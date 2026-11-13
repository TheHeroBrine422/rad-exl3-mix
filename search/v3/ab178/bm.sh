#!/bin/bash
# w142: arm B rescored from my own mix (determinism check vs w132's score of the same mix)
Q=/data/q27; W=$Q/w142; s=$1; c=$2; o=$W/scores/$s-Bm-s171d
cp $W/mix/$s-B.rad $W/mix/$s-Bm.rad || exit 1
flock $Q/locks/card$c.lock flock -s $Q/locks/ram.lock bash -c "ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$Q/rad10/home $Q/rad10/install/bin/radiance --model $W/mix/$s-Bm.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $Q/w131/slices/golden-s171d --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') score $s Bm card $c rc $? $(grep 'documents in' $o.log)" >> $W/run.log
rm -f $W/mix/$s-Bm.rad
cmp $o.json.rows $Q/w132/value/scores/$3-s171d.json.rows && echo "$s Bm rows byte-identical to w132's" >> $W/run.log || echo "$s Bm rows DIFFER from w132's" >> $W/run.log
