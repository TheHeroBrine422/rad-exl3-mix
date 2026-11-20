#!/bin/bash
# thorchain2.sh -- (after the holdout-size fix) thor: 2^20 tables K2..4, then the CPTC arms of all 22 tensors.
cd /data/work/cptcv; mkdir -p logs
CPTC_HC_SWEEPS=2 KB=2048 KN=1 bash cptc/runs/c2.sh 1048576 5 2 3 4
bash cptc/runs/c3.sh 6 5 2,3,4 cptc
echo "THORCHAIN2 DONE $(date -u +%H:%M)" >> logs/c2.done
