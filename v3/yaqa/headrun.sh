#!/bin/bash
# headrun.sh -- the row-slab LM head test (headslab.py). After gate2b.sh (both cards):
#  1. vocab permutation from the training-pool Fisher (fit + search + qad)
#  2. lm_head quantized at K2..K7 in that row order, dequantized weights + per-row errors kept (card 0, ram.lock
#     exclusive: one head quant at a time, ~5 min each)
#  3. plans: uniform / 3-4 slabs / per-block at the bytes of uniform K3 and K4
#  4. each candidate head written bf16 into a reflink copy of the bf16 container, swapped into the V2 3.00 release,
#     scored on golden-ver1 (verify pool) and golden-ml (multilingual guard); every head goes through the same bf16
#     path, so uniform vs slab is a fair comparison (V2 3.00 itself is scored alongside as the anchor)
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab PY=/data/venv-exl3/bin/python
BASE=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad BF16=${RXM_WORK}/models/final/q38-27b-bf16.rad
mkdir -p $H; cd ${RXM_WORK}
log() { echo "$(date -u +%H:%M) $*" | tee -a $H/run.log; }
until grep -q "gate2b done" $Y/gate2b.log 2>/dev/null; do sleep 60; done
[ -f $H/perm.npy ] || $PY -I $Y/headslab.py perm $Y/hs/headsens.npz $H/perm.npy >> $H/run.log 2>&1 || { log "perm FAIL"; exit 1; }
export EXL3_QT_OPTIMIZED=1
for K in 4 3 2 5 6 7; do
  [ -f $H/K$K/wq/lm_head.rowerr.npz ] && continue
  ( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock 7
    ROCR_VISIBLE_DEVICES=0 $PY -I ourq/ourq.py -K $K -o $H/K$K --only-head --device 0 --save-wq --head-perm $H/perm.npy 2>&1 \
      | grep -v -e Warning -e "cpu = _conv" ) > $H/K$K.log 2>&1
  [ -f $H/K$K/wq/lm_head.rowerr.npz ] && log "K$K head quantized: $(grep -o 'proxy [0-9.e-]*' $H/K$K.log)" || log "K$K head FAIL"
done
$PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/plan.json 2>&1 | tee -a $H/run.log
# the bf16 container's head entry must be the checkpoint's lm_head, row-major (a layout check before any overwrite)
$PY -I - $BF16 >> $H/run.log 2>&1 <<'EOF' || { log "layout check FAIL"; exit 1; }
import sys, json, numpy as np, torch
sys.path.insert(0, "/data/yaqa"); from headslab import entry
from safetensors import safe_open
off, nb, shape = entry(sys.argv[1]); V, D = shape
ck = "/data/workspace/models/qwen38-27b-bf16"
f = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
rows = [0, 1, 77777, V - 1]
with safe_open(f"{ck}/{f}", "pt", device="cpu") as h:
    sl = h.get_slice("lm_head.weight"); W = torch.cat([sl[i:i + 1] for i in rows])
R = np.memmap(sys.argv[1], np.uint16, "r", offset=off, shape=(V, D))[rows]
ok = np.array_equal(R, W.view(torch.int16).numpy().view(np.uint16))
print("layout check:", "OK" if ok else "MISMATCH", shape); sys.exit(0 if ok else 1)
EOF
Q=$H/eval.q; : > $Q
for G in "golden-ver1 corpus/ver1.jsonl" "golden-ml corpus/ml.jsonl"; do
  set -- $G; echo "hs-v2-$1 $BASE - $1 $2" >> $Q
done
for P in u3 s3-4 b3 u4 s4-4 b4; do
  D=$H/donor-$P.rad
  if [ ! -f $D.done ]; then
    cp --reflink=always $BF16 $D && $PY -I $Y/headslab.py build $H/plan.json $P $H/perm.npy $H $D >> $H/run.log 2>&1 \
      && touch $D.done || { log "$P build FAIL"; rm -f $D; continue; }
  fi
  echo "output.weight $D" > $H/$P.mix
  for G in "golden-ver1 corpus/ver1.jsonl" "golden-ml corpus/ml.jsonl"; do
    set -- $G; echo "hs-$P-$1 $BASE $H/$P.mix $1 $2" >> $Q
  done
  log "$P built, queued"
done
echo END >> $Q
./auto/evalq2.sh 0 $Q >> $H/eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $H/eval1.log 2>&1 & wait
S=${RXM_WORK}/evq/scores
for G in golden-ver1 golden-ml; do
  for B in 3 4; do
    for P in s$B-4 b$B; do
      log "$G $P vs u$B: $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/$G $S/hs-u$B-$G.f32 $S/hs-$P-$G.f32)"
    done
  done
  log "$G u4 (bf16 path) vs V2 3.00: $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/$G $S/hs-v2-$G.f32 $S/hs-u4-$G.f32)"
done
log "headrun done"
