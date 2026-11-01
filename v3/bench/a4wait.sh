#!/bin/bash
# a4wait.sh -- run a4time.sh on card 0 after d2.sh finishes (log /data/bench/a4time.log)
until grep -q "d2 done" /data/d2/d2.log 2>/dev/null; do
  pgrep -f "^/bin/bash ./d2.sh" > /dev/null || { echo "$(date -u +%H:%M) d2 not running, not done: a4 not started" >> /data/bench/a4time.log; exit 1; }
  sleep 60; done
exec 9>/root/card0.lock; flock 9
ROCR_VISIBLE_DEVICES=0 /data/bench/a4time.sh /data/bench/a4time.jsonl >> /data/bench/a4time.log 2>&1
python3 /data/bench/a4report.py /data/bench/a4time.jsonl /data/bench/speedcal-stock.jsonl >> /data/bench/a4time.log 2>&1
