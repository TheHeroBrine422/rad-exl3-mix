#!/bin/bash
# gate08.sh -- the 27B swj path on the 0.8B (4070 box), against w02 / d1106's swiglufit.py results (README "Gate").
#   A  swjfit27.py on g08-fit1 in 3 stages ([0, 8, 16, 24], bf16 boundaries), three sources in one pass:
#        k3wq  = q-k3base:wq    (swiglufit.py's input)  -> must equal w02's q-k3base_swj bit for bit (scales, swiglu.jsonl)
#        k3lv  = q-k3base:level (the 27B path: gate / up decoded from the shards in f32)
#        k15lv = d1106/q-k1.5edy:level (a fractional i8 level, as the 27B's K2.5; its --save-wq files are gone)
#   B  evalq of k3wq's wq on g08-search1 -> must equal w02's k3base_swj.f32 byte for byte
#   C  the container path: swjapply q-k3base + k3lv's scales -> decode the edited level (all 150 tensors) -> evalq;
#      D  the unedited level decoded the same way (control: decode vs --save-wq alone); pairs vs k3base / k3base_swj
# GPU steps under /data/yb/gpu.lock in two holds (A+B, then C+D), VRAM_CAP_GB 7.2. Idempotent. Log W/gate.txt.
set -uo pipefail
X=$(cd "$(dirname "$0")" && pwd); P=$X/../../../proxy2b
M=/data/models/qwen35-08b F=/data/g08-fit1 S=/data/g08-search1 Y=/data/yb W2=/data/yb/w02-fixedscale W=${W:-/data/yb/w36}
export VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p $W; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $W/gate.txt; }
cpu() { CUDA_VISIBLE_DEVICES= DEV=cpu nice -n 19 "$@"; }
ev() { [ -f $W/$1.f32 ] || python3 $P/evalq.py $M $S --wq $2 --out $W/$1.f32 2>/dev/null | tail -1 | tee -a $W/gate.txt; }
log "== gate08 start"
flock /data/yb/gpu.lock timeout 840 bash -c "
  $(declare -f log ev); X=$X P=$P M=$M S=$S W=$W
  t=\$(date +%s)
  [ -f $W/fit/k15lv/.done ] || python3 $X/swjfit27.py $M $F $W/fit --scratch $W/scr --split 0,8,16,24 --write-wq \
    --src k3wq=$Y/q-k3base:wq --src k3lv=$Y/q-k3base:level --src k15lv=$Y/d1106/q-k1.5edy:level > $W/fit.log 2>&1
  log \"A: \$(tail -1 $W/fit.log) (\$(( \$(date +%s) - t )) s)\"; t=\$(date +%s)
  ev k3wq $W/fit/k3wq/wq; log \"B: evalq \$(( \$(date +%s) - t )) s\"
  # environment control, only if A is not bit-exact: swiglufit.py itself, today (torch / kernels may have moved since 11-03)
  if ! python3 $X/cmpswj.py $W2/q-k3base_swj $W/fit/k3wq | grep -q IDENTICAL && [ ! -f $W/ref-k3base_swj/.done ]; then
    rm -rf $W/ref-k3base_swj $W/ref-k3base_swu
    python3 $X/../../../proxy2b/exp/w02-fixedscale/swiglufit.py $M $Y/q-k3base $F $W/ref-k3base_swj $W/ref-k3base_swu > $W/ref.log 2>&1
    log \"control: swiglufit.py rerun rc \$?\"; fi"
log "A k3wq vs w02 q-k3base_swj: $(python3 $X/cmpswj.py $W2/q-k3base_swj $W/fit/k3wq | tail -1)"
[ -f $W/ref-k3base_swj/.done ] && { log "control: today's swiglufit.py vs w02 (11-03): $(python3 $X/cmpswj.py $W2/q-k3base_swj $W/ref-k3base_swj | tail -1)"
  log "control: k3wq vs today's swiglufit.py: $(python3 $X/cmpswj.py $W/ref-k3base_swj $W/fit/k3wq | tail -1)"; }
log "A k3lv vs k3wq (level decode vs --save-wq):"; python3 $X/cmpswj.py $W/fit/k3wq $W/fit/k3lv | tee -a $W/gate.txt
log "A k15lv vs d1106 q-k1.5edy_swj (level vs its deleted --save-wq):"; python3 $X/cmpswj.py $Y/d1106/q-k1.5edy_swj $W/fit/k15lv | tee -a $W/gate.txt
log "B k3wq.f32 vs w02 k3base_swj.f32: $(cmp -s $W/k3wq.f32 $W2/k3base_swj.f32 && echo BYTE-IDENTICAL || echo DIFFER)"
# C / D: the container form, decoded whole (CPU), scored
cpu python3 $X/swjapply.py $Y/q-k3base $W/fit/k3lv $W/lv-k3swj | tail -1 | tee -a $W/gate.txt
[ -f $W/deq-k3swj/.done ] || { cpu python3 $X/exl3deq.py wq $W/lv-k3swj $W/deq-k3swj | tail -1 && touch $W/deq-k3swj/.done; }
[ -f $W/deq-k3/.done ] || { cpu python3 $X/exl3deq.py wq $Y/q-k3base $W/deq-k3 | tail -1 && touch $W/deq-k3/.done; }
cpu python3 $X/exl3deq.py check $W/lv-k3swj $W/fit/k3lv/wq | tail -1 | tee -a $W/gate.txt   # edited shards = level x c (f16 svh)
flock /data/yb/gpu.lock timeout 600 bash -c "$(declare -f log ev); P=$P M=$M S=$S W=$W; ev deq-k3swj $W/deq-k3swj; ev deq-k3 $W/deq-k3"
pc() { log "pair $1 -> $2: $(python3 $P/pcmp.py $S $3 $4 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
pc k3base deq-k3 $Y/k3base.f32 $W/deq-k3.f32
pc k3base_swj deq-k3swj $W2/k3base_swj.f32 $W/deq-k3swj.f32
pc k3base k3base_swj $Y/k3base.f32 $W2/k3base_swj.f32
pc deq-k3 deq-k3swj $W/deq-k3.f32 $W/deq-k3swj.f32
log "== gate08 done"
