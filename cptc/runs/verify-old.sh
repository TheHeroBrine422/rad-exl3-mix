#!/bin/bash
# verify-old.sh -- (E3_ROWSEED=1 = tensor_check.py's row sample, seed 7 + layer + 100003) do the 11-19 C3 baseline rows reproduce? 2/ssm_z mul1 K2,3,4 with the legacy search at TB=16 (as then),
# viterbi2 AVX2 at TB=16 (bit-identity claim), and viterbi2 at TB=1 (cost of the cheaper wrap search).
cd /data/work/cptcv/v3/e3; mkdir -p ../../logs
export OMP_WAIT_POLICY=passive E3_ROWSEED=1 OMP_NUM_THREADS=${T:-8} E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib
E3_TB=16 E3_V2=0 python3 -u trellis.py 2 ssm_z mul1 2,3,4 256 > ../../logs/vo-legacy-tb16.jsonl 2>&1
E3_TB=16 E3_V2=3 python3 -u trellis.py 2 ssm_z mul1 2,3,4 256 > ../../logs/vo-v3-tb16.jsonl 2>&1
E3_TB=1  E3_V2=3 python3 -u trellis.py 2 ssm_z mul1,mad 2,3,4 256 > ../../logs/vo-v3-tb1.jsonl 2>&1
echo "VERIFY DONE $(date -u +%H:%M)" >> ../../logs/c2.done
