#!/bin/bash
# gpu94.sh -- #125-01 (RULES.md amendment 1) on gpubox: real head quants C0' / B (mean-row) / T (mean-row + two-sided
# softmax-Fisher H_out, blocks path, sigma_out 0.1) at K3 / K4 with the repo's ourq (git archive in /data/w94/code/rxm)
# and the 0008 fork first on the path (/data/w94/exl3 = w92's 6242f7ae + exl3-two-sided-cpu-weight.patch: the two-sided
# path returns weight_q on the card while the head, > 5e8 weights, keeps weight_r on the CPU), then headsim94. Each quant holds
# /root/card0.lock + /root/ram.lock EXCLUSIVE (a head quant needs ~22 GB host RAM); headsim card0 + ram shared.
# Idempotent (finished donors / sims skipped). Out /data/w94/q/<arm>/K<k>, /data/w94/sim1-<set>.npz, log run94.log.
set -u
W=/data/w94 C=/data/w94/code PY=/data/venv-exl3/bin/python CK=/data/workspace/models/qwen38-27b-bf16
Q=/data/headslab3 SRC=/data/exl3frac/src EXL3=/data/w94/exl3   # = /data/w92/exl3 (6242f7ae) + exl3-two-sided-cpu-weight.patch
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HS_CKPT=$CK
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/run94.log; }
cd $W
if [ ! -s $W/hd/lm_head.safetensors ]; then
  mkdir -p $W/hd
  $PY -I - $Q/headsens.npz $Q/perm.npy $W/hd/lm_head.safetensors <<'EOF'
import sys, numpy as np, torch
from safetensors.torch import save_file
hs, pf, out = sys.argv[1:4]; F = np.load(hs)["fisher_qad_gen"][np.load(pf)]
save_file({"hout_diag": torch.from_numpy(F.astype(np.float32))}, out)
print(F.shape, float(F.min()), float(F.mean()), int((F == 0).sum()))
EOF
  log "hout file $(ls -la $W/hd/lm_head.safetensors)"
fi
hq() {   # ARM K EXTRA...
  local arm=$1 k=$2; shift 2; local d=$W/q/$arm/K$k
  [ -f $d/wq/lm_head.rowerr.npz ] && return 0
  mkdir -p $W/q/$arm; log "quant $arm K$k: waiting for card 0 + ram (exclusive)"
  ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock 7; cd /root
    echo "$(date -u +%H:%M:%S) quant $arm K$k start" >> $W/run94.log
    env PYTHONPATH=$EXL3 $PY $C/rxm/quantize/ourq.py -K $k -o $d --only-head --device 0 --save-wq \
      --head-perm $Q/perm.npy --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --codebook i8 "$@" ) > $d.log 2>&1
  [ -f $d/wq/lm_head.rowerr.npz ] && log "quant $arm K$k OK: $(cat $d/proxy.jsonl)" || { log "quant $arm K$k FAIL ($d.log)"; return 1; }
}
for k in 3 4; do
  hq c0 $k || exit 1
  hq mr $k --head-mean-row || exit 1
  hq two $k --head-mean-row --hout $W/hd --hout-blocks group --sigma-reg-out 0.1 || exit 1
done
# headsim94 over all six donors at once: fake donor labels K3 / K4 (C0'), K13 / K14 (B), K23 / K24 (T)
mkdir -p $W/qs
for k in 3 4; do ln -sfn $W/q/c0/K$k $W/qs/K$k; ln -sfn $W/q/mr/K$k $W/qs/K1$k; ln -sfn $W/q/two/K$k $W/qs/K2$k; done
python3 -c "
import json
p = {f'u{a}{k}': {'K_blocks': [float(f'{a}{k}')] * 1940, 'avg_bits': float(k)} for a in ('', '1', '2') for k in (3, 4)}
json.dump({'Ks': sorted(float(n[1:]) for n in p), 'plans': p}, open('$W/plan1.json', 'w'))"
for s in search qad; do
  [ -s $W/sim1-$s.npz ] && continue
  log "headsim1 $s: waiting for card 0"
  ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
    HS_TAU_PLANS=none $PY -I $C/headsim94.py $W/dump/$s $W/qs $Q/perm.npy $W/sim1-$s.npz $W/plan1.json ) > $W/sim1-$s.log 2>&1
  log "headsim1 $s rc $?"
done
log "GPU94 DONE"
