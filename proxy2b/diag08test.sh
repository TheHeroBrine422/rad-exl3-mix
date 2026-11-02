#!/bin/bash
# diag08test.sh -- #33-17: the 27B diagonal collector (v3/yaqa/collect.py YAQA_DIAG) run on the 0.8B on one GPU:
#   1. gate (2 seqs): the per-linear diagonal vs a whole fp32 G
#   2. 40 sequences straight (snapshots 20, 40)
#   3. 20 sequences, then a resumed run to 40: must equal (2) exactly (resume is bit-exact)
#   4. halves.json written at each snapshot
# Data: the calib golden's qad-pool windows (as on the 27B), pools-v3.
export VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/data/c08work; mkdir -p $W/challenge/golden; ln -sfn /data/challenge/calib-meta $W/challenge/golden/calib
export RXM_WORK=$W RXM_CKPT=/data/models/qwen35-08b YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
export YAQA_FP32=1 YAQA_DIAG=1 YAQA_SEED=0
C=/workspace/rad-exl3-mix/v3/yaqa/collect.py O=/data/c08test; rm -rf $O; mkdir -p $O; cd /tmp
DIAG_CHECK=23.mlp.down_proj YAQA_GATE=1 python3 $C $O/gate 2 2048 > $O/gate.log 2>&1; grep -E "gate check|^seq|Error" $O/gate.log | tail -4
YAQA_SNAP=20,40 YAQA_SAVE_EVERY=1000 python3 $C $O/straight 40 2048 > $O/straight.log 2>&1; tail -n 2 $O/straight.log
YAQA_SNAP=20 YAQA_SAVE_EVERY=20 python3 $C $O/resumed 20 2048 > $O/res1.log 2>&1
YAQA_SNAP=40 YAQA_SAVE_EVERY=1000 python3 $C $O/resumed 40 2048 > $O/res2.log 2>&1; grep -E "resuming|DONE" $O/res2.log
python3 - <<'PY'
import glob, json, os, torch
from safetensors import safe_open
O = "/data/c08test"; worst = 0; n = 0
for f in glob.glob(f"{O}/straight/hdiag/*.safetensors"):
    g = f.replace("/straight/", "/resumed/")
    with safe_open(f, "pt") as a, safe_open(g, "pt") as b:
        x, y = a.get_tensor("hout_diag"), b.get_tensor("hout_diag"); worst = max(worst, float((x - y).abs().max())); n += 1
print(f"resume check: {n} linears, max |straight - resumed| = {worst:.3e} ({'BIT-EXACT' if worst == 0 else 'DIFFERS'})")
for s in ("hdiag-snap20", "hdiag-snap40"):
    h = json.load(open(f"{O}/straight/{s}/halves.json")); print(f"{s}: halves cos min {h['min']:.4f} p5 {h['p5']:.4f} median {h['median']:.4f}")
PY
echo DIAG08TEST DONE
