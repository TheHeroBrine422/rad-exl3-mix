#!/bin/bash
# vtrun.sh -- wait for the head decodes, then vtsim.py on card 1 (card lock, then ram.lock shared)
cd /data/vt
until grep -q HEADDECDONE headdec.log; do sleep 10; done
exec 5>/root/card1.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
ROCR_VISIBLE_DEVICES=1 /data/venv-exl3/bin/python -I vtsim.py /root/e1b/dumphead /data/vt /data/vt/rank-sq-counts.npy \
  /data/workspace/models/qwen38-27b-bf16 /data/vt/vtsim2.json > vtsim2.txt 2> vtsim2.err
echo "VTSIMDONE rc=$?" >> vtsim2.err
