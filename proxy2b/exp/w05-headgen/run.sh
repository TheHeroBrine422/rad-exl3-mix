#!/bin/bash
# run.sh -- w05-headgen (ranked.md E4): LM-head statistics from scored (gen) rows only. gpubox ONLY.
#  0. CPU: is the slab ranking gen-only already? (headslab.py ranks by fisher_qad_gen: report the overlap with _all)
#  1. headx.py, BOTH CARDS (~30-60 min): bf16 forward of qad docs (2M tokens, 8k cap) -> gen-row and all-position head
#     input Hessians of the same documents; gen-row dump of fit-pool docs (65,536 rows, doc ids) -- also w05-head2s's
#     second scoring set
#  2. CPU gate (hgate.py): gen-H vs all-H quant, error on gen rows, in-sample; < 2% at K2 and K3 -> STOP here
#  3. K3 head donors (one card, host RAM exclusive): vocabulary order, mean-row, sigma 0.005 (w05-head2s's `one`
#     settings), --hin hin-gen vs --hin hin-all, rotation seeds 0-2 (6 quants)
#  4. headarms on the gen-row dump (primary: the scored rows, held-out docs) and the calib-A dump; judge.py (rule in
#     README); the published calib H (w05-head2s `one`, if present) reported beside them
: ${RXM_WORK:=/root} ${CARD:=0}; export RXM_WORK
H=$(dirname $(readlink -f $0)); Y=/data/yaqa O=/data/yaqa/w05-headgen PY=/data/venv-exl3/bin/python
CK=/data/workspace/models/qwen38-27b-bf16 SRC=/data/exl3frac/src DUMP=$RXM_WORK/e1b/dumphead G=$RXM_WORK/challenge/golden/calib
POOLS=$RXM_WORK/challenge/pools-v3.json SENS=$Y/hs/headsens.npz SPERM=$Y/headslab2/perm.npy HS2=/data/yaqa/w05-head2s
A2=$H/../w05-head2s/src HX=$O/hx
mkdir -p $O/q $O/sim; cd $RXM_WORK
log() { echo "$(date -u +%F\ %T) $*" | tee -a $O/result.txt; }
# ---- 0. ranking (CPU, seconds)
[ -f $O/rank.txt ] || $PY -I - $SENS <<'EOF' | tee $O/rank.txt | tee -a $O/result.txt
import sys, numpy as np
z = np.load(sys.argv[1]); g, a = z["fisher_qad_gen"], z["fisher_qad_all"]
og, oa = np.argsort(-g), np.argsort(-a)
for n in (128, 1280, 20000): print(f"top {n} rows: gen / all overlap {len(set(og[:n]) & set(oa[:n])) / n:.3f}")
r = lambda x: np.argsort(np.argsort(-x)); live = (g > 0) | (a > 0)
print(f"Spearman (rows with mass) {np.corrcoef(r(g)[live], r(a)[live])[0, 1]:.4f}; headslab.py already ranks by fisher_qad_gen")
EOF
# ---- 1. headx (both cards)
if [ ! -f $HX/headx.json ]; then
  ( exec 5>$RXM_WORK/card0.lock 6>$RXM_WORK/card1.lock 7>$RXM_WORK/ram.lock; flock 5; flock 6; flock -s 7
    ROCR_VISIBLE_DEVICES=0,1 $PY -I $H/src/headx.py $CK $G $POOLS $HX --hpools qad --dpools fit --tok 2e6 --cap 8192 \
      --dump 65536 --per-doc 1024 --calib-hess $RXM_WORK/challenge/hessians/calib/lm_head.input.safetensors ) > $O/headx.log 2>&1
  [ -f $HX/headx.json ] && log "headx: $(cat $HX/headx.json | tr -d '\n ')" || { log "headx FAIL: $(tail -n 3 $O/headx.log | tr '\n' ' ')"; exit 1; }
fi
# ---- 2. gate (CPU, ~5 min at 8 threads)
QF=$($PY -c "import exllamav3.modules.quant.exl3_lib.quantize as q; print(q.__file__)" 2>/dev/null | tail -1)
[ -f $O/gate.json ] || $PY -I $H/src/hgate.py $HX $CK $QF $O/gate.json --sens $SENS > $O/gate.log 2>&1
log "gate: $(tail -n 3 $O/gate.log | tr '\n' ' ')"
$PY -c "import json, sys; sys.exit(0 if json.load(open('$O/gate.json'))['go'] else 1)" || { log "W05-HEADGEN STOP at the gate"; exit 0; }
# ---- 3. donors
export ROCR_VISIBLE_DEVICES=$CARD EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A L=([gen]="" [all]="")
for s in 0 1 2; do for h in gen all; do
  d=$O/q/h$h-K3-s$s
  if [ ! -f $d/wq/lm_head.rowerr.npz ]; then
    ( exec 9>$RXM_WORK/card$CARD.lock 7>$RXM_WORK/ram.lock; flock 9; flock 7
      timeout 45m $PY -I ourq/ourq.py -K 3 -o $d --only-head --device 0 --save-wq --qtf2 --frac-ext $SRC \
        --head-mean-row --sigma-reg 0.005 --seed-offset $s --hin $HX/hin-$h ) > $d.log 2>&1
    [ -f $d/wq/lm_head.rowerr.npz ] && log "donor h$h s$s OK" || log "donor h$h s$s FAIL: $(tail -n 2 $d.log | tr '\n' ' ')"
  fi
  L[$h]="${L[$h]} h$h-K3-s$s=$d"
done; done
CAL=""; for s in 0 1 2; do [ -f $HS2/q/one-K3-s$s/wq/lm_head.rowerr.npz ] && CAL="$CAL hcalib-K3-s$s=$HS2/q/one-K3-s$s"; done
ARMS="${L[all]} ${L[gen]} $CAL"        # the first arm (hall seed 0) is headarms' reference
# ---- 4. score + judge (control = hall: the same documents, all positions)
( exec 9>$RXM_WORK/card$CARD.lock 7>$RXM_WORK/ram.lock; flock 9; flock -s 7
  $PY -I $A2/headarms.py $HX/dump $CK $O/sim/gen $ARMS --sens-perm $SPERM > $O/sim/gen.log 2>&1
  $PY -I $A2/headarms.py $DUMP $CK $O/sim/calibA $ARMS --sens-perm $SPERM > $O/sim/calibA.log 2>&1 )
$PY -I $A2/judge.py $O/sim/gen $HX/dump K3 hall "hgen hcalib" $O/judge-gen.json --half all --thr 0.97 --guard 1.1 \
  --sens-perm $SPERM | sed 's/^/gen dump: /' | tee -a $O/result.txt
$PY -I $A2/judge.py $O/sim/calibA $DUMP K3 hall "hgen hcalib" $O/judge-calibA.json --half all --thr 0.97 --guard 1.1 \
  --sens-perm $SPERM | sed 's/^/calib-A dump: /' | tee -a $O/result.txt
log "W05-HEADGEN DONE"
