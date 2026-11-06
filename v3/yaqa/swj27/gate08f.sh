#!/bin/bash
# gate08f.sh -- the --fused grid kernel (swjtri.py) on the 4070, after gate08.sh: one GPU hold (/data/yb/gpu.lock).
#   bench  swjbench.py at the 27B's shape (J 17,408, T 2,048): torch vs fused time per position, the kernel's difference
#   fit    swjfit27.py --fused on g08-fit1, source q-k3base:wq, same split -> vs gate08's exact k3wq (cmpswj)
#   ev     only if any c_g / c_u differs: evalq of the fused wq, paired with the exact k3wq.f32
set -uo pipefail
X=$(cd "$(dirname "$0")" && pwd); P=$X/../../../proxy2b
M=/data/models/qwen35-08b F=/data/g08-fit1 S=/data/g08-search1 Y=/data/yb W=${W:-/data/yb/w36}
export VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd /tmp; log() { echo "$(date -u +%T) $*" | tee -a $W/gate.txt; }
[ -f $W/fit/k3wq/.done ] || { log "gate08f: run gate08.sh first"; exit 1; }
log "== gate08f start"
flock /data/yb/gpu.lock timeout 600 bash -c "
  $(declare -f log); W=$W
  python3 $X/swjbench.py 17408 2048 2>&1 | grep -v Warn | tail -1 | tee -a $W/gate.txt
  t=\$(date +%s)
  [ -f $W/fitF/k3wq/.done ] || python3 $X/swjfit27.py $M $F $W/fitF --scratch $W/scrF --split 0,8,16,24 --write-wq --fused \
    --src k3wq=$Y/q-k3base:wq > $W/fitF.log 2>&1
  log \"fused fit: \$(grep -h 'fused check' $W/fitF.log | head -1 | cut -c10-); \$(grep -h 'tok/s)' $W/fitF.log | sed 's/.*(\([0-9]* tok\/s\)).*/\1/' | tr '\n' ' ')(\$(( \$(date +%s) - t )) s; exact: \$(grep -h 'tok/s)' $W/fit.log | sed 's/.*(\([0-9]* tok\/s\)).*/\1/' | tr '\n' ' ')3 sources)\"
  if ! python3 $X/cmpswj.py $W/fit/k3wq $W/fitF/k3wq | grep -q 'scales identical'; then
    [ -f $W/k3wqF.f32 ] || python3 $P/evalq.py $M $S --wq $W/fitF/k3wq/wq --out $W/k3wqF.f32 2>/dev/null | tail -1 | tee -a $W/gate.txt; fi"
log "fused vs exact k3wq:"; python3 $X/cmpswj.py $W/fit/k3wq $W/fitF/k3wq | tee -a $W/gate.txt
[ -f $W/k3wqF.f32 ] && log "pair k3wq -> k3wqF: $(python3 $P/pcmp.py $S $W/k3wq.f32 $W/k3wqF.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"
log "== gate08f done"
