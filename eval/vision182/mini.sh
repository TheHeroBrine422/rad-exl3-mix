#!/bin/bash
# mini.sh -- (w156) smoke: bf16 single card layer_offload --kld-record of 4 docs (3 image, 1 text), then K2.5 --kld-ref
W=/data/q27/w156; Q=/data/q27; I=$W/rad/install; H=$W/rad/home; L=$Q/locks; CARD=7
mkdir -p $W/mini; cd $W/mini
( head -n 3 $W/corpus/corpus.jsonl; sed -n 451p $W/corpus/corpus.jsonl; sed -n 300p $W/corpus/corpus.jsonl ) > corpus.jsonl
( flock 5; flock -s 4
  rm -rf gold
  cd $Q && ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$H timeout 1800 $I/bin/radiance --model $W/packs/K4.rad --tp 1 \
    --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 6144 --mm-max-patches 4096 \
    --kld-record $W/mini/gold --kld-corpus $W/mini/corpus.jsonl --kld-top 512 > $W/mini/gold.log 2>&1; echo "gold rc $?"
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$H timeout 1800 $I/bin/radiance --model $W/packs/K2.5.rad --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --mm-max-patches 4096 --kld-ref $W/mini/gold --kld-out $W/mini/k25.json > $W/mini/k25.log 2>&1; echo "k25 rc $?"
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$H timeout 1800 $I/bin/radiance --model $W/packs/K2.5.rad --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $W/mini/gold --kld-out $W/mini/k25-notower.json > $W/mini/k25-notower.log 2>&1; echo "k25 no tower rc $?"
) 5>$L/card$CARD.lock 4>$L/ram.lock
