#!/bin/bash
# b19wait.sh -- B19 K1 decode bit-compare on card 1 right after the early gate 2e verdict (seconds). Log /data/exl3frac/e3/b19.log
until grep -q "GATE 2E" /data/yaqa/overnight.log 2>/dev/null; do
  pgrep -f "^/bin/bash ./overnight.sh" > /dev/null || exit 1; sleep 30; done
cd /data/exl3frac/e3
( exec 9>/root/card1.lock; flock 9
  ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=1 timeout 600 /data/venv-exl3/bin/python -I b19k1.py /data/workspace/references/td-SC_1.40bpw_H3 /data/exl3frac/e3/exl3_decode.so 4 ) > b19.log 2>&1
