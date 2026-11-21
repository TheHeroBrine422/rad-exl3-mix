#!/bin/bash
# one tensor of layer 3 through a build: t1.sh BUILD CARD KEY HNAME K [args]
B=$1; CARD=$2; KEY=$3; HN=$4; K=$5; shift 5
cd /data/q27/w210; . ./env.sh; export ROCR_VISIBLE_DEVICES=$CARD OMP_NUM_THREADS=8
P=model.language_model.layers.3
KEYS=$(echo $KEY | sed "s/[^,]*/$P.&/g")
flock /data/q27/locks/card$CARD.lock $B/exl3v3_tool tensor --key $KEYS --hname $P.$HN --K $K --ckpt /data/models/qwen38-27b-bf16 --hess /data/q27/challenge/hessians/calib "$@" 2> >(grep -v "file not found\|hipblaslt\|hipModuleLoad\|rocblaslt" >&2)
