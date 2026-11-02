#!/bin/bash
# headremake.sh -- the final head donors, run in ladder2.sh's exclusive RUN: slot (card $CARD and exclusive host RAM are
# already held by the slot: this script takes NO locks -- re-locking from a new file descriptor would block on itself).
#  1. settings test: K3 and K5 x {mean-row on: default damping, sigma 0.01, 0.005} vs the existing default donors;
#     priced by headsim (uniform heads). Rules (fixed 2026-11-01; mean-row must be the same in every donor, or slab
#     rows from different donors stop sharing one logit shift): mean-row on if it gains on the geometric mean of K3 / K5;
#     damping one value if K3 and K5 agree, else every donor runs all three and
#     keeps the lowest headsim uniform KL.
#  2. donors (plain mul1 codebook: the head never takes the int8 path; fractional on qtf2): K1-3 by quarters, 3-6 by
#     quarters, 6-8 by halves.
#  3. floor curves (HS_FLOOR_ALL: one plan per tail floor, headsim chooses): C donors K1-3.25 budgets 1.5-3.5,
#     A K3-6 budgets 3.5-6, B K5-8 budgets 5.5-7.75; 1/8-bit budgets.
# Each donor has a 45 min timeout (K7.5 hung once): a FAIL is logged and the curves price without it.
# STOP_AFTER=settings: step 1 only (dry run outside the slot; the slot skips what is done).
# Out /data/yaqa/headfinal/ (K*/ donors, curve-*.json, settings.json), log run.log.
: ${RXM_WORK:=/root} ${CARD:=0}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 O=/data/yaqa/headfinal PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16 SRC=/data/exl3frac/src
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=$CARD EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build
hq() {   # DIR K EXTRA... : one head donor (skips a finished one)
  local d=$1 k=$2; shift 2
  [ -f $d/wq/lm_head.rowerr.npz ] && return 0
  mkdir -p $(dirname $d)
  timeout 45m $PY -I ourq/ourq.py -K $k -o $d --only-head --device 0 --save-wq --head-perm $H/perm.npy --qtf2 --frac-ext $SRC "$@" \
    > $d.log 2>&1
  [ -f $d/wq/lm_head.rowerr.npz ] && log "donor $d OK" || { log "donor $d FAIL"; return 1; }
}
price() {   # QDIR PLAN OUT, env HS_* for the planner
  $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $1 $2 >> $O/run.log 2>&1 && \
  $PY -I $Y/headsim.py $DUMP $1 $H/perm.npy $2 $CK $3 > $3.log 2>&1
  [ -s $3 ] || { log "headsim FAIL $3"; return 1; }
}
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
export HS_TRAIN=qad

# ---- 1. settings
S=$O/settings
for K in 3 5; do mkdir -p $S/default; ln -sfn $H/K$K $S/default/K$K; done
for V in mr mr-d0.01 mr-d0.005; do
  X="--head-mean-row"; case $V in *d0.01) X="$X --sigma-reg 0.01";; *d0.005) X="$X --sigma-reg 0.005";; esac
  for K in 3 5; do hq $S/$V/K$K $K $X; done
done
for V in default mr mr-d0.01 mr-d0.005; do
  [ -s $S/$V.json ] || HS_FLOOR=3 HS_KS=3,5 HS_BUDGETS=4 price $S/$V $S/$V-plan.json $S/$V.json
done
python3 - $S $O/settings.json <<'EOF' | tee -a $O/run.log
import json, sys
S, out = sys.argv[1:3]
kl = {V: {k: json.load(open(f"{S}/{V}.json"))["plans"][f"u{k}"]["kl"] for k in (3, 5)} for V in ("default", "mr", "mr-d0.01", "mr-d0.005")}
mr = {k: kl["mr"][k] < kl["default"][k] for k in (3, 5)}
damp = {k: min((kl[v][k], v) for v in ("mr", "mr-d0.01", "mr-d0.005"))[1] for k in (3, 5)}
import math
g = math.sqrt(kl["mr"][3] / kl["default"][3] * kl["mr"][5] / kl["default"][5])
r = {"kl": kl, "mean_row": mr, "mean_row_geo_ratio": g, "use_mean_row": g < 1, "best_damping_variant": damp,
     "damping_agree": damp[3] == damp[5]}
r["sigma"] = {"mr": None, "mr-d0.01": 0.01, "mr-d0.005": 0.005}[damp[3]] if r["damping_agree"] else "sweep"
json.dump(r, open(out, "w"), indent=1); print("settings:", json.dumps(r))
EOF
MR=$(python3 -c "import json; print(int(json.load(open('$O/settings.json'))['use_mean_row']))")
SIG=$(python3 -c "import json; print(json.load(open('$O/settings.json'))['sigma'])")
log "settings: mean-row $MR, sigma $SIG"
# headsim's donor grouping (HS_MAXK) must reproduce the pre-change pricing of the same donors exactly
[ -s /data/yaqa/headset/default.json ] && python3 -c "
import json, sys; a, b = (json.load(open(f))['plans'] for f in sys.argv[1:3])
print('headsim regression', 'OK' if all(a[n]['kl'] == b[n]['kl'] for n in a) else f'DIFFERS {a} {b}')" \
  /data/yaqa/headset/default.json $S/default.json | tee -a $O/run.log
[ "$STOP_AFTER" = settings ] && exit 0

# ---- 2. donors
DONORS="1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5 5.25 5.5 5.75 6 6.5 7 7.5 8"
X=""; [ "$MR" = 1 ] && X="--head-mean-row"
for K in $DONORS; do
  if [ "$SIG" = sweep ]; then   # all three, keep the lowest exact uniform KL
    for sg in default 0.01 0.005; do
      Y2=""; [ $sg != default ] && Y2="--sigma-reg $sg"
      hq $O/sweep/$sg/K$K $K $X $Y2
    done
    best=""; bk=9
    for sg in default 0.01 0.005; do
      [ -f $O/sweep/$sg/K$K/wq/lm_head.rowerr.npz ] || continue
      Q=$O/sweep/q-$sg-K$K; mkdir -p $Q; ln -sfn $O/sweep/$sg/K$K $Q/K$K
      HS_FLOOR=0 HS_KS=$K HS_BUDGETS=$K price $Q $Q/plan.json $Q/sim.json || continue
      v=$(python3 -c "import json; print(json.load(open('$Q/sim.json'))['plans']['u$K']['kl'])")
      python3 -c "import sys; sys.exit(0 if float('$v') < float('$bk') else 1)" && { bk=$v; best=$sg; }
    done
    [ -n "$best" ] && ln -sfn $O/sweep/$best/K$K $O/K$K && log "K$K: damping $best (KL $bk)"
  else
    Y2=""; [ "$SIG" != None ] && Y2="--sigma-reg $SIG"
    hq $O/K$K $K $X $Y2
  fi
done

# ---- 3. floor curves
export HS_FLOOR_ALL=1
HS_FLOOR=1 HS_KS=1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25 HS_BUDGETS=$(bud 1.5 3.5) price $O $O/curve-C-plan.json $O/curve-C.json
HS_FLOOR=3 HS_KS=3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6 HS_BUDGETS=$(bud 3.5 6) price $O $O/curve-A-plan.json $O/curve-A.json
HS_FLOOR=5 HS_KS=5,5.25,5.5,5.75,6,6.5,7,7.5,8 HS_BUDGETS=$(bud 5.5 7.75) price $O $O/curve-B-plan.json $O/curve-B.json
log "headremake done"
