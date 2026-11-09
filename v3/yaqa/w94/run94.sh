#!/bin/bash
# run94.sh -- w94 #124 on gpubox (RULES.md): head-input dumps on card 0, G1, plans, headsim94 on both dumps, and the
# real donors' rows on hq94's blocks. Every GPU step waits for /root/card0.lock, then /root/ram.lock shared (lock
# order card, ram), and releases both after it. Idempotent: finished outputs are skipped on a rerun.
# Out /data/w94/: dump/<set>/ (headx94), gate-<set>.json, curve.json + plan.json, sim-<set>.npz, real-K<k>.npy, run94.log
set -u
W=/data/w94 C=/data/w94/code PY=/data/venv-exl3/bin/python CK=/data/workspace/models/qwen38-27b-bf16
G=/root/challenge/golden/calib Q=/data/headslab3
export ROCR_VISIBLE_DEVICES=0 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 OMP_NUM_THREADS=4 HS_CKPT=$CK
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/run94.log; }
cardrun() { ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; "$@" ); }
cd $W

# plans (CPU, seconds): the corrected-rowerr plan set of #34-02 and a 1/8-bit curve over today's K3-K7 donors
cp -n $Q/plan.json $W/plan.json
if [ ! -s $W/curve.json ]; then
  B=$(python3 -c "print(','.join(f'{3 + i / 8:g}' for i in range(1, 33)))")
  HS_TRAIN=qad HS_FLOOR=3 HS_FLOOR_ALL=1 HS_BUDGETS=$B $PY -I $C/headslab.py plan $Q/headsens.npz $Q/perm.npy $Q \
    $W/curve.json.part > $W/curve.log 2>&1 && mv $W/curve.json.part $W/curve.json
  [ -s $W/curve.json ] && log "curve plans OK" || { log "curve plans FAIL"; exit 1; }
fi

if [ ! -f $W/dump/qad/done.json ] || [ ! -f $W/dump/search/done.json ]; then
  log "dump: waiting for card 0"
  cardrun bash -c "echo \$(date -u +%H:%M:%S) dump start >> $W/run94.log; $PY -I $C/headx94.py $CK $G $C/docs94.json $W/dump > $W/headx94.log 2>&1"
  log "dump rc $? $(cat $W/dump/*/done.json 2>/dev/null | tr '\n' ' ')"
fi
for s in search qad; do
  [ -f $W/dump/$s/done.json ] || { log "dump $s missing"; exit 1; }
  if [ ! -s $W/gate-$s.json ]; then
    cardrun $PY -I $C/gate94.py $W/dump/$s $G $CK $W/gate-$s.json 1000000 >> $W/gate.log 2>&1
    log "G1 $s: $(cat $W/gate-$s.json)"
  fi
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$W/gate-$s.json'))['pass'] else 1)" || { log "G1 FAIL $s: stop"; exit 1; }
done
for s in search qad; do
  if [ ! -s $W/sim-$s.npz ]; then
    log "headsim $s: waiting for card 0"
    cardrun $PY -I $C/headsim94.py $W/dump/$s $Q $Q/perm.npy $W/sim-$s.npz $W/plan.json $W/curve.json > $W/sim-$s.log 2>&1
    log "headsim $s rc $?"
  fi
done
# the real donors' rows on hq94's 64 blocks (CPU, ram.lock shared): K3 / K4 wq, f16, sensitivity order
for k in 3 4; do
  [ -s $W/real-K$k.npy ] && continue
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY -I - $Q/K$k/wq/lm_head.safetensors $C/blocks94.json $W/real-K$k.npy <<'EOF'
import json, sys, numpy as np, torch
from safetensors import safe_open
src, bj, out = sys.argv[1:4]; b = json.load(open(bj))["all"]
rows = (torch.tensor(b)[:, None] * 128 + torch.arange(128)).reshape(-1)
with safe_open(src, "pt") as f: W = f.get_tensor("weight")
np.save(out + ".part.npy", W[rows].numpy()); import os; os.replace(out + ".part.npy", out)
EOF
  )
  log "real rows K$k $(ls -la $W/real-K$k.npy 2>&1)"
done
log "RUN94 DONE"
