#!/bin/bash
# blkorder27.sh -- (gpubox, card 0, the lead runs it; ~2 min) rxq's #117 block order (quant.block_order, on by default
# since 11-08) re-makes w84's p0perm readers bit for bit: the 9 readers of w84's noop gate (L0 ssm_qkv / ssm_z / gate /
# up, L3 q / k / v / gate / up) quantized by `rxq quant` with w84's p0 settings (rxq levelargs of the control
# /data/v3ladder2/K2.5, quant.tile_cd off, as w84.sh) and the block order on, compared tensor by tensor (trellis, suh,
# svh, mul1) with w84's /data/w84/p0/q-perm (made by blkorder84.py via w84q.py). CPU twin: tests/check_blkorder84.py.
# Staging: the rad-exl3-mix tree (pipeline/ with this change) under $CODE, as w84.sh. Locks: card0.lock, ram.lock shared.
#   CODE=/data/w86/code bash blkorder27.sh            -> $OUT/result.txt, last line "BLKORDER27 PASS n/9" or FAIL
set -u
: ${RXM_WORK:=/root}; CODE=${CODE:-/data/w86/code}; OUT=${OUT:-/data/w86/blk27}; PY=${PY:-/data/venv-exl3/bin/python}
BASE=${BASE:-/data/v3ladder2/K2.5}; PERM=${PERM:-/data/w84/p0/q-perm}
MODEL=${MODEL:-/data/workspace/models/qwen38-27b-bf16} HESS=${HESS:-/root/challenge/hessians/calib}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build OMP_NUM_THREADS=4
RX=(env PYTHONPATH=$CODE/pipeline RXQ_RXM=$CODE RXQ_EXL3_SRC=${SRC:-/data/exl3frac/src} $PY)
mkdir -p $OUT; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $OUT/result.txt; }
L=model.language_model.layers
KEYS9=$L.0.linear_attn.in_proj_qkv,$L.0.linear_attn.in_proj_z,$L.0.mlp.gate_proj,$L.0.mlp.up_proj,$L.3.self_attn.q_proj,$L.3.self_attn.k_proj,$L.3.self_attn.v_proj,$L.3.mlp.gate_proj,$L.3.mlp.up_proj
X=$("${RX[@]}" -m rxq levelargs $BASE) || { log "FAIL rxq levelargs $BASE"; exit 1; }
X=${X/--set quant.block_order.enable=false/}                              # w84's arm: the order on
X="$X --set quant.tile_cd.enable=false"
log "settings: $X"
( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
  nice -n 5 "${RX[@]}" -m rxq quant --keys $KEYS9 $X --ckpt $MODEL --hess $HESS -o $OUT/q-perm9 --device 0 --no-prof ) > $OUT/quant.log 2>&1 \
  || { log "FAIL quant ($OUT/quant.log)"; exit 1; }
"${RX[@]}" - $PERM $OUT/q-perm9 $KEYS9 <<'EOF' | tee -a $OUT/result.txt
import json, sys, torch
from safetensors import safe_open
a, b, keys = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
ok = 0
for k in keys:
    with safe_open(f"{a}/t-{k}.safetensors", "pt") as x, safe_open(f"{b}/t-{k}.safetensors", "pt") as y:
        same = sorted(x.keys()) == sorted(y.keys()) and all(torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys())
    rec = [json.loads(l) for l in open(f"{b}/log.jsonl") if json.loads(l)["key"] == k][-1]
    print(f"{'same' if same else 'DIFF'} {k} (rxq log block_order {rec.get('block_order')})"); ok += same
print(f"BLKORDER27 {'PASS' if ok == len(keys) else 'FAIL'} {ok}/{len(keys)}")
EOF
