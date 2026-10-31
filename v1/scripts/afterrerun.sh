#!/bin/bash
# afterrerun.sh -- once plain6.00's capped rerun has its result line: delete the sweep-only SC 4.00 variants and the
# capped goldens (2026-10-29), then resume the calib golden download. Log: /root/afterrerun.log
until grep -q '"tag": "plain6.00-e6"' /root/fullfinal.jsonl && [ ! -d /root/ff-plain6.00-e6 ]; do sleep 30; done
rm -f /root/models/final/q38-27b-sc4.00-e3.rad /root/models/final/q38-27b-sc4.00-e5.rad /root/models/final/q38-27b-sc4.00-e6.rad
rm -rf /root/golden-cap200k /root/golden-cap160k
echo "deleted; $(df -h /data | tail -1)"
cd /root; s=$(date +%s)
HF_TOKEN=$(cat /root/.hf_token) hf download StillDeadcode/qwen3.8-27b-quant-challenge --repo-type dataset \
  --include "corpus/*" "golden/calib/*" "tools/*" README.md --local-dir /root/challenge > /root/dl-calib.log 2>&1
echo "download rc=$? $(( $(date +%s) - s ))s $(du -sh /root/challenge | cut -f1); $(df -h /data | tail -1)"
