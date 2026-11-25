#!/bin/bash
# read231.sh NAME CKPTDIR CARD GOLDEN [SIZE] (box2) -- w231: one served read of a checkpoint on any judge slice, no results.tsv row (the equivalence gate's reads):
# the queue's swap222.py (released 2.50 container + the checkpoint's scales) and qscore.sh (plain radiance --kld-ref, bf16 KV) under the card's lock; scores
# QROOT/scores/NAME.json(.rows); the patched container is deleted. Compare reads with qad226/jl226.py QROOT CTL NAME --golden GOLDEN.
set -u
N=$1 CKD=$2 CARD=$3 G=$4 SIZE=${5:-2.50}
source /data/q27/w231/code/v3/qadq/qenv.sh box2
[ -e $CKD/.done ] || { echo "$N: checkpoint $CKD not finished"; exit 3; }
[ -s $QROOT/scores/$N.json ] && { echo "$N done"; exit 0; }
eval "exec {fd}>$(CARDLOCK $CARD)"; flock $fd; export QCARD_LOCKED=1
P=$QROOT/patched/$N.rad; mkdir -p $QROOT/patched
$PY $HERE/swap222.py $IN/rel/host-emb/qwen3.8-27b-exl3-v3-${SIZE}bpw.rad $IN/lv/$SIZE $CKD $P --verify > $QROOT/logs/swap-$N.log 2>&1 || { echo "swap failed"; exit 4; }
$HERE/qscore.sh box2 $N $P $CARD $G; rc=$?; rm -f $P; exit $rc
