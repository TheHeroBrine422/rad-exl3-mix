#!/bin/bash
# run232x.sh (box2: cd /data/q27/w232x; setsid nohup code/v3/qad232/run232x.sh < /dev/null > /dev/null 2>&1 &) -- w232 part C: the V3.1 final-read x part QROOT/slices/golden-v31x
# (sel232.py --holdout --mix jgap-mix.json: xgap HOLDOUT docs not in the gap judge, <= 64k tokens, whole). Same chain232.py (TP2 bf16 recording, top-512, bf16 KV, genmask181, check232) with its own root
# /data/q27/w232x; lane A blocks on the card-0,1 locks (polled every 2 s) until the xg-fin chain (run232.sh) releases them. only-card0/1 = NONE is kept in place while this runs.
# Restartable (same command).
W=/data/q27/w232x; Q=/data/q27/queue; cd $W; mkdir -p logs
echo NONE > $Q/only-card0; echo NONE > $Q/only-card1
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') run232x start" >> $W/chain.log
W232_ROOT=$W W232_CODE=/data/q27/w232/code/v3/qad232 W232_FINAL=$Q/slices/golden-v31x W232_CORPUS_OUT=$Q/slices/golden-v31x.jsonl W232_MASK_OUT=$W/xg-v31x.u8 W232_POLL=2 \
  TZ=America/Chicago /data/q27/venv/bin/python -I /data/q27/w232/code/v3/qad232/chain232.py >> $W/logs/chain.out 2>&1 &
CP=$!
while kill -0 $CP 2>/dev/null; do [ -e $Q/only-card0 ] || echo NONE > $Q/only-card0; [ -e $Q/only-card1 ] || echo NONE > $Q/only-card1; sleep 2; done
wait $CP; RC=$?; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') run232x: chain exited rc $RC" >> $W/chain.log
rm -f $Q/only-card0 $Q/only-card1
