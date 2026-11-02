#!/bin/bash
# yaqaFact8.sh -- #33-19 (YAQA08.md amendment 12): diagonal H_out collected on our own agentic traces (pisess v1)
# vs calib qad windows (#33-18's eB), and their 50/50 mix; 0.8B, K3 / K2, 3 seeds. Waits for yaqaFact7.sh.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 C=/data/yb/c19
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact8.txt; }
while pgrep -f "^/bin/bash /workspace/rad-exl3-mix/proxy2b/[y]aqaFact7.sh" > /dev/null; do sleep 60; done
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
cd /tmp
if [ ! -f $C/hdiag-snap1024/model.language_model.layers.23.mlp.down_proj.safetensors ]; then
  W=/data/c19work; mkdir -p $W/challenge/golden; ln -sfn /data/pisess/v1/meta $W/challenge/golden/calib
  RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=/data/pisess/v1/pools-all-qad.json YAQA_FP32=1 YAQA_DIAG=1 YAQA_SEED=0 \
  YAQA_SNAP=1024 YAQA_SAVE_EVERY=64 python3 /workspace/rad-exl3-mix/v3/yaqa/collect.py $C 1024 2048 > $O/c19.log 2>&1
  grep -q "^DONE" $O/c19.log || { log "COLLECT FAIL"; exit 1; }
fi
[ -f $C/mix/.done ] || python3 - <<'PY'
import glob, os, torch
from safetensors import safe_open
from safetensors.torch import save_file
a, b, d = "/data/yb/c18/hdiag-snap1024", "/data/yb/c19/hdiag-snap1024", "/data/yb/c19/mix"; os.makedirs(d, exist_ok=True)
cs = []
for f in glob.glob(f"{a}/*.safetensors"):
    n = os.path.basename(f)
    with safe_open(f, "pt") as x, safe_open(f"{b}/{n}", "pt") as y: u, v = x.get_tensor("hout_diag").double(), y.get_tensor("hout_diag").double()
    cs.append((float((u * v).sum() / (u.norm() * v.norm())), n))
    save_file({"hout_diag": ((u / u.mean() + v / v.mean()) / 2).float()}, f"{d}/{n}")
cs.sort(); import statistics
print(f"cos(qad, pisess) per linear: min {cs[0][0]:.4f} ({cs[0][1][28:]}), median {statistics.median(c for c, _ in cs):.4f}")
open(f"{d}/.done", "w").close()
PY
log "$(python3 -c 'print(1)' >/dev/null; grep -h "cos(qad" /data/logs/yaqaFact8.log | tail -1)"
for s in 0 1 2; do sfx=$([ $s = 0 ] && echo "" || echo "_s$s"); for k in 3 2; do
  q k${k}pis$sfx $k --hout $C/hdiag-snap1024 --sigma-reg-out 0.01 --seed-offset $s
  q k${k}pmix$sfx $k --hout $C/mix --sigma-reg-out 0.01 --seed-offset $s
done; done
log FACT8 DONE
