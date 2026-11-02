#!/bin/bash
# night27.sh -- the unattended 27B night (2026-11-02/03), on iceman, both cards (user's arrangement):
#   1. hdiag27.sh 512: diagonal H_out collection (Sketch B, sum dy^2, sum ||x||^2 dy^2; fp32; 2-seq gate first)
#   2. card 0, after the collection: two K3 trunk quants with the ladder's settings (ladder2.sh: i8, qtf2, scaled
#      Hessian, damping 0.025 / input 0.005):
#        a) + diagonal H_out from sum dy^2 (512 seqs), output damping 0.01    -> /data/v3night/K3-dy/K3
#        b) no H_out, rotation seed offset 1 (the 27B quant-noise floor, ranked A7) -> /data/v3night/K3-s1/K3
#      against the ladder's K3 (seed 0, /data/v3ladder2/K3). Evals and the swap test: morning, with a pass rule.
# Stops on any failure (log /data/v3night/night.log); nothing here deletes data. Stop: touch /data/yaqa/hdiag/STOP.
: ${RXM_WORK:=/root}; export RXM_WORK
N=/data/v3night; mkdir -p $N; L=$N/night.log; log() { echo "$(date -u +%H:%M) $*" | tee -a $L; }
PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
log "start: collection"
bash /root/ourq/hdiag27.sh 512 >> $L 2>&1
[ -f /data/yaqa/hdiag/hdiag-snap512-dy/model.language_model.layers.63.mlp.down_proj.safetensors ] || { log "no 512 dy snapshot: stop"; exit 1; }
log "collection done: halves $($PY -c "import json; h=json.load(open('/data/yaqa/hdiag/hdiag-snap512/halves.json')); print(h['median'], h['p5'], h['min'])")"
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
X="--qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --codebook i8"
quant() { tag=$1; shift
  [ $(ls $N/$tag/K3/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && { log "$tag: exists"; return 0; }
  log "$tag: start"; s=$(date +%s)
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    $PY -I ourq/ourq.py -K 3 -o "$N/$tag/K{K}" --device 0 $X "$@" >> $N/$tag.log 2>&1 )
  n=$(ls $N/$tag/K3/q-L*.safetensors 2>/dev/null | wc -l)
  [ "$n" = 64 ] && log "$tag: done in $(( ($(date +%s) - s) / 60 )) min" || { log "$tag: FAIL ($n layers; $N/$tag.log)"; return 1; }
}
quant K3-dy --hout /data/yaqa/hdiag/hdiag-snap512-dy --sigma-reg-out 0.01 || exit 1
quant K3-s1 --seed-offset 1 || exit 1
log "NIGHT DONE"
