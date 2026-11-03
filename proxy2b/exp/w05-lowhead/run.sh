#!/bin/bash
# run.sh -- w05-lowhead (ranked.md E1): row-slab LM heads at low budgets (2.25-3.75 bits / row) for the 2.0 / 2.5 mixes.
# ICEMAN ONLY (27B head). Steps, each skipped when its output exists:
#  1. donors K2..K4 by quarters in sensitivity order (headslab2/perm.npy), mean-row, sigma_reg 0.005 (head-settings:
#     K3's winner): linked from headremake.sh's /data/yaqa/headfinal/K<k> when that exists (its own damping sweep),
#     else quantized here (card $CARD + host RAM exclusive, as headremake.sh)
#  2. plans: every 1/8 budget 2.25..3.75 x every tail floor >= K2 (HS_FLOOR_ALL), + uniform plans; headsim on the
#     calib-A head dump (held out from the qad Fisher ranking)
#  3. pickcurve.py -> $O/headcurve.json (guard rule); knap.sh (CPU) prices it in knap4 at 2.0 / 2.5 / 3.0
#  4. whole-model transfer check (card $CARD): bf16 heads spliced into V2 2.50 (and 2.00), the same bf16 path for all:
#     u3 (uniform K3 donor), slab3 (the curve's budget-3.0 plan, equal bytes to u3), the knap4 2.5 pick (2.50 base) and
#     2.0 pick (2.00 base), scored on golden-s2k (search pool) and golden-ml (multilingual guard); paired.py vs u3.
: ${RXM_WORK:=/root} ${CARD:=0}; export RXM_WORK
H=$(dirname $(readlink -f $0)); Y=/data/yaqa O=/data/yaqa/w05-lowhead PY=/data/venv-exl3/bin/python
CK=/data/workspace/models/qwen38-27b-bf16 SRC=/data/exl3frac/src DUMP=$RXM_WORK/e1b/dumphead
SENS=$Y/hs/headsens.npz PERM=$Y/headslab2/perm.npy HF=/data/yaqa/headfinal
BASE=/data/release/qwen3.8-27b-exl3-v2-2.50bpw.rad BASE20=/data/release/qwen3.8-27b-exl3-v2-2.00bpw.rad BF16=$RXM_WORK/models/final/q38-27b-bf16.rad
KS="2 2.25 2.5 2.75 3 3.25 3.5 3.75 4"
mkdir -p $O/q; cd $RXM_WORK
log() { echo "$(date -u +%F\ %T) $*" | tee -a $O/result.txt; }
export ROCR_VISIBLE_DEVICES=$CARD EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---- 1. donors
for K in $KS; do
  d=$O/q/K$K
  [ -f $d/wq/lm_head.rowerr.npz ] && continue
  if [ -f $HF/K$K/wq/lm_head.rowerr.npz ]; then ln -sfn $(readlink -f $HF/K$K) $d; log "K$K: headfinal donor"; continue; fi
  ( exec 9>$RXM_WORK/card$CARD.lock 7>$RXM_WORK/ram.lock; flock 9; flock 7
    timeout 45m $PY -I ourq/ourq.py -K $K -o $d --only-head --device 0 --save-wq --head-perm $PERM --qtf2 --frac-ext $SRC \
      --head-mean-row --sigma-reg 0.005 ) > $d.log 2>&1
  [ -f $d/wq/lm_head.rowerr.npz ] && log "K$K donor OK" || log "K$K donor FAIL: $(tail -n 2 $d.log | tr '\n' ' ')"
done
# ---- 2. plans + headsim
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
DON=$(for K in $KS; do [ -f $O/q/K$K/wq/lm_head.rowerr.npz ] && echo -n "$K,"; done); DON=${DON%,}
if [ ! -s $O/curve.json ]; then
  HS_TRAIN=qad HS_FLOOR=2 HS_FLOOR_ALL=1 HS_KS=$DON HS_BUDGETS=$(bud 2.25 3.75) \
    $PY -I $Y/headslab.py plan $SENS $PERM $O/q $O/curve-plan.json >> $O/plan.log 2>&1 || { log "plan FAIL"; exit 1; }
  ( exec 9>$RXM_WORK/card$CARD.lock 7>$RXM_WORK/ram.lock; flock 9; flock -s 7
    $PY -I $Y/headsim.py $DUMP $O/q $PERM $O/curve-plan.json $CK $O/curve.json ) > $O/curve.json.log 2>&1
  [ -s $O/curve.json ] && log "curve priced (donors $DON)" || { log "headsim FAIL"; exit 1; }
fi
# ---- 3. curve -> knap4
[ -s $O/headcurve.json ] || python3 $H/src/pickcurve.py $O/curve.json $O/headcurve.json 1.3 | tee -a $O/result.txt
REPO=${REPO:-$RXM_WORK/rad-exl3-mix}     # a checkout with data/v3 (knap4 inputs); else run knap.sh on the VM / colo
[ -s $O/knap/hc.jsonl ] || { [ -d $REPO/data/v3 ] && bash $H/knap.sh $O/headcurve.json $O/knap $REPO | tee -a $O/result.txt; }
[ -s $O/knap/hc.jsonl ] || { log "no knap4 result ($O/knap/hc.jsonl): run knap.sh elsewhere, copy it back, rerun"; exit 0; }
# ---- 4. whole-model transfer check
P3=$(python3 -c "import json; print(json.load(open('$O/headcurve.json'))['budgets'].get('3', {}).get('plan', ''))")
pick() {   # SIZE -> the slab plan knap4 picked for the head at that size ("" if none)
python3 - $O/knap/hc.jsonl $O/headcurve.json $1 <<'EOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if abs(json.loads(l)["target"] - float(sys.argv[3])) < 1e-9][0]
hc = [k for k in r["param_share"] if k.startswith("ours-hcK")]
print(json.load(open(sys.argv[2]))["budgets"][hc[0][8:]]["plan"] if hc else "")
EOF
}
P25=$(pick 2.5) P20=$(pick 2.0)
[ -n "$P3" ] || { log "no budget-3.0 plan in the curve: whole-model check skipped"; exit 0; }
$PY -I - $BF16 >> $O/build.log 2>&1 <<'EOF' || { log "layout check FAIL (bf16 container head != checkpoint)"; exit 1; }
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
sys.exit(0 if np.array_equal(R, W.view(torch.int16).numpy().view(np.uint16)) else 1)
EOF
Q=$O/eval.q; : > $Q
for P in u3 $P3 $P25 $P20; do
  D=$O/head-$P.rad
  if [ ! -f $D.done ]; then
    cp --reflink=always $BF16 $D && $PY -I $Y/headslab.py build $O/curve-plan.json $P $PERM $O/q $D >> $O/build.log 2>&1 \
      && touch $D.done || { log "$P build FAIL"; rm -f $D; continue; }
  fi
  echo "output.weight $D" > $O/$P.mix
done
for G in "golden-s2k corpus/s2k.jsonl" "golden-ml corpus/ml.jsonl"; do set -- $G
  for P in u3 $P3 $P25; do echo "w05lh-25-$P-$1 $BASE $O/$P.mix $1 $2" >> $Q; done
  [ -n "$P20" ] && for P in u3 $P20; do echo "w05lh-20-$P-$1 $BASE20 $O/$P.mix $1 $2" >> $Q; done
done
echo END >> $Q
./auto/evalq2.sh $CARD $Q >> $O/eval.log 2>&1      # takes card$CARD.lock and ram.lock itself
S=$RXM_WORK/evq/scores
for G in golden-s2k golden-ml; do
  for P in $P3 $P25; do [ -f $S/w05lh-25-$P-$G.f32 ] && \
    log "2.50 base, $G: $P vs u3: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/$G $S/w05lh-25-u3-$G.f32 $S/w05lh-25-$P-$G.f32)"; done
  [ -n "$P20" ] && [ -f $S/w05lh-20-$P20-$G.f32 ] && \
    log "2.00 base, $G: $P20 vs u3: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/$G $S/w05lh-20-u3-$G.f32 $S/w05lh-20-$P20-$G.f32)"
done
python3 -c "
import json; c = json.load(open('$O/curve.json'))['plans']
print('headsim predicted head-only KL: u3', c['u3']['kl'], '$P3', c['$P3']['kl'], 'diff', c['$P3']['kl'] - c['u3']['kl'])" | tee -a $O/result.txt
log "W05-LOWHEAD DONE"
