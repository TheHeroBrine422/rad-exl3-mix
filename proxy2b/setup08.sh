#!/bin/bash
# setup08.sh -- the 0.8B's golden (the same fit-pool slice as g2b-fit1) and its calib input Hessians. Logs /data/logs/.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f /data/g08-fit1/kld.json ] || python3 $P/mkgold.py $M /data/challenge/calib-meta /data/g08-fit1 $(cat /data/slice-fit1.txt) > /data/logs/g08.log 2>&1
python3 $P/check_score.py $M /data/g08-fit1 /data/challenge/tools/score.py 4 2>&1 | tail -2
[ -d /data/h08/calib ] || python3 $P/hcollect.py $M /data/challenge/calib-meta /data/h08/calib --layers 0-23 --head 2>&1 | tr '\r' '\n' | grep -E "DONE|peak|Error" 
echo SETUP08 DONE
