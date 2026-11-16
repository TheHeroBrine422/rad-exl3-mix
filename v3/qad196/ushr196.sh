#!/bin/bash
# ushr196.sh LEVEL TAG (w182, #196 U) -- the un-shrink factors (unshrink196.py) of one virtual level on box2 CPU: one
# process per linear type (CPU only, no card), then one merged file data/U-TAG.safetensors (+ its stats json).
# Restartable: a finished type's file is skipped. Level TAG: 25 = vl/inc2.5m, 40 = vl/inc4.0m.
set -u
LEVEL=$1 TAG=$2; DEV=${DEV:-cpu}     # DEV=cuda:N: the GPU (card lock held by the caller)
cd /data/q27/w182 && source env.sh
D=$W/data; mkdir -p $D
pids=()
for T in ${TYPES:-attn_output ssm_out ffn_down attn_v ssm_qkv}; do
  [ -f $D/u$TAG-$T.safetensors ] && continue
  OMP_NUM_THREADS=8 nice -n 5 $PY $CODE/v3/qad196/unshrink196.py $D/u$TAG-$T.safetensors --level $LEVEL --model $MODEL \
    --hessians /data/q27/challenge/hessians/calib --only $T --dev $DEV > $D/u$TAG-$T.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || echo "process $p failed"; done
$PY - "$D" "$TAG" <<'PY'
import sys, json, glob
from safetensors.torch import load_file, save_file
D, TAG = sys.argv[1], sys.argv[2]
out, stats = {}, {}
for f in sorted(glob.glob(f"{D}/u{TAG}-*.safetensors")):
    if f.endswith(".stats.json"): continue
    out.update(load_file(f))
    try: stats.update(json.load(open(f + ".stats.json")))
    except Exception: pass
save_file(out, f"{D}/U{TAG}.safetensors.part"); import os; os.replace(f"{D}/U{TAG}.safetensors.part", f"{D}/U{TAG}.safetensors")
json.dump(stats, open(f"{D}/U{TAG}.stats.json", "w"), indent=1)
print("merged", len(out), "linears")
PY
