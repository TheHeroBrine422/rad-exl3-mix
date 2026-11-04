#!/bin/bash
# chunkfloor.sh -- 0.8B bf16 vs its own golden at prefill chunk 0 (whole) / 2048 / 512 / 64; queued behind d1104.
while pgrep -f "^/bin/bash /workspace/rad-exl3-mix/proxy2b/exp/d1104/[r]un.sh" > /dev/null; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=7.2; cd /tmp
python3 -I /workspace/rad-exl3-mix/proxy2b/chunkfloor.py /data/models/qwen35-08b /data/g08-search1 0 2048 512 64 > /data/yb/chunkfloor.jsonl 2> /data/yb/chunkfloor.err
