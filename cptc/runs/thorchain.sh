#!/bin/bash
# thorchain.sh -- thor's share of the 11-20 CPTC rerun: 2^20 tables K2..4 (then the CPTC arms of all 22 tensors), the base arms
# of the 3 ffn_down tensors box2 does not take, and the old-row reproduction check.
cd /data/work/cptcv; mkdir -p logs
T=8 bash cptc/runs/verify-old.sh &
LIST="27 ffn_down|3 ffn_down|2 ffn_down" bash cptc/runs/c3.sh 3 5 2,3,4 base &
CPTC_HC_SWEEPS=2 KB=2048 KN=1 bash cptc/runs/c2.sh 1048576 5 2 3 4
bash cptc/runs/c3.sh 6 5 2,3,4 cptc
wait; echo "THORCHAIN DONE $(date -u +%H:%M)" >> logs/c2.done
