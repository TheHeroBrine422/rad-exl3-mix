#!/bin/bash
# hest.sh -- #33-29 (4070): the three 27B H_out diagonal estimators (Σdy², Sketch B's diagonal, xw = Σ||x||²dy²) at
# the same sample, quantizer-level proxy on hconv's 3 layers at K2.5. Train on the even 128-sequence blocks up to
# SNAP, score E_out under each estimator's odd-block mean (3 references: a reference built from one estimator favours
# it, so a ranking only counts if it holds under all three). Needs the snapshots synced by hestsync.sh (VM).
# Shares /data/yb/gpu.lock; idempotent. Rule: README.md "#33-29".
R=/workspace/rad-exl3-mix V=$R/v3/yaqa/hconv O=/data/yb/hconv M=/data/models/qwen38-27b-bf16 HS=/data/challenge/dl/hessians/calib
SNAP=${SNAP:-1664} LAYERS="10 39 50" K=2.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/hest.txt; }
for e in dy B xw; do s=$([ $e = B ] && echo "" || echo $e)
  [ -d $O/hest-$e/train-$SNAP ] || python3 -I $V/mkhalf.py $O/snaps $O/hest-$e "$s" | tee -a $O/hest.txt; done
W=/data/p27b; SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
q() { arm=$1; shift; for L in $LAYERS; do
  [ -f $O/q-$arm/q-L$(printf %02d $L).safetensors ] && continue
  ( mkdir -p /tmp/hconv-run && cd /tmp/hconv-run && OURQ_REFIT_CPU_FALLBACK=1 PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=$M RXM_TEMPLATE=$W/template.json \
    TORCH_EXTENSIONS_DIR=/data/torch_ext flock /data/yb/gpu.lock python3 $R/quantize/ourq.py -K $K -o $O/q-$arm --device 0 --layers $L-$L --codebook i8 --qtf2 \
    --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq "$@" ) >> $O/q-$arm.log 2>&1 || log "QUANT FAIL $arm L$L"
done; }
for e in dy B xw; do q est$e$SNAP --hout $O/hest-$e/train-$SNAP --sigma-reg-out 0.01; done
ARMS="none=$O/q-none/wq"; for e in dy B xw; do ARMS="$ARMS est$e=$O/q-est$e$SNAP/wq"; done
for r in dy B xw; do
  flock /data/yb/gpu.lock python3 -I $V/hscore.py $M $HS $O/hest-$r/test $ARMS > $O/hest-score-$r.jsonl 2>> $O/hest.err
  log "reference: $r odd blocks"
  python3 - $O/hest-score-$r.jsonl <<'PY' | tee -a $O/hest.txt
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]; base = {r["key"]: r for r in R if r["arm"] == "none"}
by = collections.defaultdict(list)
for r in R: by[r["arm"]].append(r)
for a in sorted(by):
    rs = [r for r in by[a] if r["key"] in base]; w = sum(r["params"] for r in rs)
    print(f"{a:>8} {len(rs):3d} tensors  E_out/none {sum(r['params'] * r['E_out'] / base[r['key']]['E_out'] for r in rs) / w:.4f}")
PY
done
log "HEST DONE"
