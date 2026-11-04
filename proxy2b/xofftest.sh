#!/bin/bash
# xofftest.sh -- YAQA_XOFF=1 (inputs parked on the host) must give bit-identical diagonals to the default, 0.8B, 24 seqs.
M=/data/models/qwen35-08b O=/data/yb/xofftest W=/data/c08work; mkdir -p $O $W/challenge/golden; ln -sfn /data/challenge/calib-meta $W/challenge/golden/calib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=7.2
C="RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json YAQA_FP32=1 YAQA_DIAG=1 YAQA_DIAG_EXTRA=1 YAQA_GENMASK=1 YAQA_SEED=0 YAQA_SNAP=24 YAQA_SAVE_EVERY=24"
cd /tmp
for a in off on; do [ -d $O/$a/hdiag-snap24-dy ] && continue; rm -rf $O/$a
  env $C YAQA_XOFF=$([ $a = on ] && echo 1 || echo 0) python3 /workspace/rad-exl3-mix/v3/yaqa/collect.py $O/$a 24 2048 > $O/$a.log 2>&1
  grep "^seq 23\|^DONE" $O/$a.log | tail -2 | cut -c1-200; done
python3 - <<'PY'
import glob
from safetensors import safe_open
O = "/data/yb/xofftest"; n = 0; bad = []
for sub in ("hdiag-snap24", "hdiag-snap24-dy", "hdiag-snap24-xw"):
    for f in sorted(glob.glob(f"{O}/off/{sub}/*.safetensors")):
        with safe_open(f, "pt") as a, safe_open(f.replace("/off/", "/on/"), "pt") as b:
            n += 1
            if not (a.get_tensor("hout_diag") == b.get_tensor("hout_diag")).all(): bad.append(f.split("/")[-2:])
print(f"XOFF test: {n} files, {len(bad)} differ {bad[:3]} -> {'PASS' if n and not bad else 'FAIL'}")
PY
