#!/bin/bash
# w210 box2: synthetic expert batches (DS4.1-F shapes) on one card. syn.sh CARD TAG [tool args]
CARD=$1; TAG=$2; shift 2
cd /data/q27/w210; . ./env.sh; export ROCR_VISIBLE_DEVICES=$CARD OMP_NUM_THREADS=8; mkdir -p out/syn
[ -s out/syn/$TAG.jsonl ] && exit 0
flock /data/q27/locks/card$CARD.lock build/exl3v3_tool batch "$@" > out/syn/$TAG.jsonl.part 2> out/syn/$TAG.err && mv out/syn/$TAG.jsonl.part out/syn/$TAG.jsonl
