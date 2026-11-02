#!/bin/bash
# hrun.sh -- the 2B's calib input Hessians, three passes of 8 layers (the fp32 sums of 24 layers do not fit beside the
# model in the 4070's free memory). Documents <= 32k tokens, run whole. Out /data/h2b/calib, log /data/logs/hrun.log.
P=/workspace/rad-exl3-mix/proxy2b
for r in 0-7 8-15 16-23; do
  python3 $P/hcollect.py /data/models/qwen35-2b /data/challenge/calib-meta /data/h2b/calib --layers $r 2>&1 | tr '\r' '\n' | grep -v -i "warn\|Loading" | tail -3
done
ls /data/h2b/calib | wc -l; du -sh /data/h2b/calib; echo HRUN DONE
