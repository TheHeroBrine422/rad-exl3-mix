#!/bin/bash
# run.sh -- w05-head2s (ranked.md E2): two-sided LDLQ on the 27B LM head with its exact softmax-Fisher diagonal
# (headsens.npz fisher_qad_gen), block-diagonal path (quantize-hout-bd.patch + ourq-hout-bd.patch, MERGED first).
# ICEMAN ONLY (the 0.8B head is tied). One card, host RAM exclusive per head quant (~22-26 GB), as headremake.sh.
#   stage 1  K3 seed 0: two-sided in vocabulary order (two-v) and strided order (two-s) at sigma_out 0.01 / 0.1 / 1;
#            sigma_out per family picked on the SELECTION half of the calib-A dump (rule in README)
#   stage 2  K3 seeds 0-2: one (control, one-sided, vocabulary order), two-v*, two-s*, hd-v (one-sided + Fisher output
#            scales, ourq --hout-diag); one-s (strided one-sided) seed 0 as a diagnostic; judged on the REPORT half
#   stage 3  K5 seeds 0-2: one vs the stage-2 winner (only if one passed at K3)
# Every arm: mean-row, sigma_reg 0.005, mul1 / qtf2 (headremake.sh's donor settings). Scored by src/headarms.py
# (exact head-only KL, 32,768 calib-A head inputs; also E4's gen-row dump if it exists). Idempotent: finished donors
# (wq/lm_head.rowerr.npz) and scored arms (.kl.f32) are skipped. Results appended to $O/result.txt.
: ${RXM_WORK:=/root} ${CARD:=0}; export RXM_WORK
H=$(dirname $(readlink -f $0)); Y=/data/yaqa O=/data/yaqa/w05-head2s PY=/data/venv-exl3/bin/python
CK=/data/workspace/models/qwen38-27b-bf16 SRC=/data/exl3frac/src DUMP=$RXM_WORK/e1b/dumphead
SENS=$Y/hs/headsens.npz SPERM=$Y/headslab2/perm.npy GDUMP=/data/yaqa/w05-headgen/hx/dump
mkdir -p $O; cd $RXM_WORK
log() { echo "$(date -u +%F\ %T) $*" | tee -a $O/result.txt; }
export ROCR_VISIBLE_DEVICES=$CARD EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
grep -q -- "--hout-bd" ourq/ourq.py && $PY -c "from exllamav3.modules.quant.exl3_lib.quantize import ldlq_2hess_bd" \
  || { log "patches not merged (ourq --hout-bd / quantize.ldlq_2hess_bd): stop"; exit 1; }
# ---- inputs: the Fisher vector (vocabulary order) and the strided order (block b = sensitivity ranks b, b + nb, ...)
[ -f $O/fisher/lm_head.safetensors ] || $PY -I - $SENS $SPERM $O <<'EOF'
import sys, os, numpy as np, torch
from safetensors.torch import save_file
sens, sp, O = sys.argv[1:4]
F = np.load(sens)["fisher_qad_gen"]; p = np.load(sp); nb = len(p) // 128
os.makedirs(f"{O}/fisher", exist_ok=True)
save_file({"hout_diag": torch.from_numpy(F.astype(np.float32))}, f"{O}/fisher/lm_head.safetensors", metadata={"src": "fisher_qad_gen"})
np.save(f"{O}/strided.npy", p.reshape(128, nb).T.reshape(-1).astype(np.int64))
print("fisher + strided order written")
EOF
# ---- one head donor: NAME K SEED EXTRA... (exclusive card + host RAM; 60 min timeout, a hang is logged as FAIL)
hq() {
  local n=$1 k=$2 s=$3; shift 3; local d=$O/q/$n-K$k-s$s
  [ -f $d/wq/lm_head.rowerr.npz ] && return 0
  ( exec 9>$RXM_WORK/card$CARD.lock 7>$RXM_WORK/ram.lock; flock 9; flock 7
    timeout 60m $PY -I ourq/ourq.py -K $k -o $d --only-head --device 0 --save-wq --qtf2 --frac-ext $SRC \
      --head-mean-row --sigma-reg 0.005 --seed-offset $s "$@" ) > $d.log 2>&1
  [ -f $d/wq/lm_head.rowerr.npz ] && log "donor $n K$k s$s OK ($(grep -o '"s": [0-9.]*' $d/proxy.jsonl | tail -1))" \
    || { log "donor $n K$k s$s FAIL: $(tail -n 2 $d.log | tr '\n' ' ')"; return 1; }
}
arm() {   # NAME K SEED -> headarms spec
  local n=$1 k=$2 s=$3; local d=$O/q/$n-K$k-s$s
  case $n in two-s*|one-s) echo "$n-K$k-s$s=$d:$O/strided.npy";; *) echo "$n-K$k-s$s=$d";; esac
}
score() {   # TAG ARMS... : headarms on the calib-A dump (and the gen-row dump when E4 made one)
  local t=$1; shift
  ( exec 9>$RXM_WORK/card$CARD.lock 7>$RXM_WORK/ram.lock; flock 9; flock -s 7
    $PY -I $H/src/headarms.py $DUMP $CK $O/sim/calibA "$@" --sens-perm $SPERM > $O/sim/$t-calibA.log 2>&1
    [ -f $GDUMP/rows.i64 ] && $PY -I $H/src/headarms.py $GDUMP $CK $O/sim/gen "$@" --sens-perm $SPERM > $O/sim/$t-gen.log 2>&1 )
  log "scored $t: $(grep -c KL $O/sim/$t-calibA.log) arm lines"
}
mkdir -p $O/q $O/sim
TWO="--hout $O/fisher --hout-bd"
# ---- stage 1
for so in 0.01 0.1 1; do
  hq two-v$so 3 0 $TWO --sigma-reg-out $so
  hq two-s$so 3 0 $TWO --sigma-reg-out $so --head-perm $O/strided.npy
done
hq one 3 0
score stage1 $(arm one 3 0) $(for so in 0.01 0.1 1; do arm two-v$so 3 0; arm two-s$so 3 0; done)
[ -f $O/pick.json ] || $PY -I - $O/sim/calibA/report.json $O/pick.json <<'EOF' | tee -a $O/result.txt
import json, sys
r = json.load(open(sys.argv[1]))["halves"]["sel"]; pick = {}
for fam in ("two-v", "two-s"):     # lowest selection-half KL with guard KL <= 1.3 x the control's
    c = {n: v for n, v in r.items() if n.startswith(fam) and n.endswith("-K3-s0") and (v.get("guard_ratio") or 9) <= 1.3}
    pick[fam] = min(c, key=lambda n: c[n]["kl"]).split("-K3")[0][len(fam):] if c else None
    print(f"stage 1 {fam}: sigma_out {pick[fam]} ({ {n: (round(v['ratio'], 3), round(v.get('guard_ratio') or 0, 2)) for n, v in r.items() if n.startswith(fam)} })")
json.dump(pick, open(sys.argv[2], "w"))
EOF
SV=$($PY -c "import json; print(json.load(open('$O/pick.json'))['two-v'] or '')")
SS=$($PY -c "import json; print(json.load(open('$O/pick.json'))['two-s'] or '')")
# ---- stage 2
A2=""
for s in 0 1 2; do
  hq one 3 $s; A2="$A2 $(arm one 3 $s)"
  hq hd-v 3 $s --hout-diag $O/fisher; A2="$A2 $(arm hd-v 3 $s)"
  [ -n "$SV" ] && { [ $s = 0 ] || hq two-v$SV 3 $s $TWO --sigma-reg-out $SV; A2="$A2 $(arm two-v$SV 3 $s)"; }
  [ -n "$SS" ] && { [ $s = 0 ] || hq two-s$SS 3 $s $TWO --sigma-reg-out $SS --head-perm $O/strided.npy; A2="$A2 $(arm two-s$SS 3 $s)"; }
done
hq one-s 3 0 --head-perm $O/strided.npy; A2="$A2 $(arm one-s 3 0)"
# the slab comparator at equal bytes (w05-lowhead's donors and curve, if they exist): budget 3.0's chosen plan
LH=/data/yaqa/w05-lowhead
[ -f $LH/headcurve.json ] && P3=$($PY -c "import json; print(json.load(open('$LH/headcurve.json'))['budgets'].get('3', {}).get('plan', ''))") \
  && [ -n "$P3" ] && A2="$A2 slab3=plan:$LH/curve-plan.json:$P3:$LH/q:$SPERM"
score stage2 $A2
$PY -I $H/src/judge.py $O/sim/calibA $DUMP K3 one "two-v$SV two-s$SS hd-v" $O/judge-K3.json --sens-perm $SPERM | tee -a $O/result.txt
[ -f $GDUMP/rows.i64 ] && $PY -I $H/src/judge.py $O/sim/gen $GDUMP K3 one "two-v$SV two-s$SS hd-v" $O/judge-K3-gen.json \
  --half all --sens-perm $SPERM | sed 's/^/gen dump: /' | tee -a $O/result.txt
$PY -I - $O/sim/calibA/report.json <<'EOF' | tee -a $O/result.txt      # the slab comparator, report half (one seed)
import json, sys
r = json.load(open(sys.argv[1]))["halves"]["rep"]
if "slab3" in r: print("slab3 (w05-lowhead budget 3.0) vs one-K3-s0:", round(r["slab3"]["ratio"], 3), "guard x", round(r["slab3"].get("guard_ratio") or 0, 2))
EOF
# ---- stage 3 (K5), only for a family that passed at K3
W=$($PY -c "import json; print(json.load(open('$O/judge-K3.json'))['winner'] or '')" 2>/dev/null)
if [ -n "$W" ]; then
  X3=""; case $W in two-v*) X="$TWO --sigma-reg-out $SV";; two-s*) X="$TWO --sigma-reg-out $SS --head-perm $O/strided.npy";; hd-v) X="--hout-diag $O/fisher";; esac
  for s in 0 1 2; do hq one 5 $s; hq $W 5 $s $X; X3="$X3 $(arm one 5 $s) $(arm $W 5 $s)"; done
  score stage3 $X3
  $PY -I $H/src/judge.py $O/sim/calibA $DUMP K5 one "$W" $O/judge-K5.json --thr 0.95 --sens-perm $SPERM | tee -a $O/result.txt
fi
log "W05-HEAD2S DONE"
