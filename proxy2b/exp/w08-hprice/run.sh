#!/bin/bash
# run.sh -- w08-hprice (ranked F1) on the 4070: (0) the ourq-hprice.patch gate, 3 two-layer K3 quants (~3 min GPU);
# (1) after w10-hout-attrib has run: per-family check of the two-sided price (CPU, ~4 min, no GPU). Idempotent.
# Results: /data/yb/w08-hprice/result.txt.
R=/workspace/rad-exl3-mix P=$R/proxy2b X=$P/exp/w08-hprice M=/data/models/qwen35-08b
O=/data/yb/w08-hprice W10=/data/yb/w10-hout-attrib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
while pgrep -f "quantize/[o]urq.py|proxy2b/[e]valq.py" > /dev/null; do sleep 60; done   # one job at a time on the card
# ---- 0. gate: REF = q2b.sh as is; OFF / ON = the same with the patched ourq (a copy beside a v3 symlink, so its
#      --frac-ext lookups resolve as in the repo), ON with --hprice. Layers 0-1, K3, base settings.
Q=/tmp/w08q; if [ ! -f $Q/quantize/ourq.py ]; then
  mkdir -p $Q/quantize && cp $R/quantize/ourq.py $Q/quantize/ && ln -sfn $R/v3 $Q/v3
  # the patch builds on w04-blockgain/rowerr.patch (the same rowerr fix): apply that first unless merged
  grep -q "def load_w" $Q/quantize/ourq.py || (cd $Q && patch -p1 --quiet < $P/exp/w04-blockgain/rowerr.patch) || { log "GATE: rowerr.patch does not apply"; exit 1; }
  (cd $Q && patch -p1 --quiet < $X/ourq-hprice.patch) || { log "GATE: patch does not apply"; exit 1; }
fi
q2bp() {   # q2b.sh with OURQ=$1 (the rest of q2b.sh unchanged)
  local OURQ=$1; shift; local O_=$1; shift
  W=/data/p08; mkdir -p $W/challenge/hessians; ln -sfn /data/h08/calib $W/challenge/hessians/calib
  python3 -c "import json; c=json.load(open('$M/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$W/template.json', 'w'))"
  mkdir -p /tmp/q2b-run && cd /tmp/q2b-run
  PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=$M RXM_TEMPLATE=$W/template.json TORCH_EXTENSIONS_DIR=/data/torch_ext \
    python3 $OURQ -K 3 -o $O_ --device 0 --layers 0-1 --codebook i8 --qtf2 --frac-ext /data/src/exllamav3/exllamav3/exllamav3_ext \
    --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq "$@"; cd - > /dev/null; }
if ! grep -q "^.* GATE" $O/result.txt 2>/dev/null; then
  [ -f $O/gref/q-L01.safetensors ] || $P/q2b.sh 08 3 i8 $O/gref --layers 0-1 > $O/gref.log 2>&1
  [ -f $O/goff/q-L01.safetensors ] || q2bp $Q/quantize/ourq.py $O/goff > $O/goff.log 2>&1
  [ -f $O/gon/q-L01.safetensors ] || q2bp $Q/quantize/ourq.py $O/gon --hprice /data/yb/dv8192/hout > $O/gon.log 2>&1
  log "$(CUDA_VISIBLE_DEVICES= python3 $X/gatecmp.py $O/gref $O/goff $O/gon | tr '\n' ' ')"
fi
# ---- 1. per family (needs w10's subset arms k{3,2}h-<family>{,_s1,..}); rowerr2 with w10's estimator (Σdy², c18)
if grep -q "DONE" $W10/result.txt 2>/dev/null && [ ! -f $O/f1fam.json ]; then
  [ -s $O/rowerr2-edy.jsonl ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python3 $X/rowerr2.py $M /data/h08/calib \
    /data/yb/c18/hdiag-snap1024-dy $(for t in k3base k3base_s1 k3base_s2 k2base k2base_s1 k2base_s2 k3edy k3edy_s1 k3edy_s2 \
    k2edy k2edy_s1 k2edy_s2; do echo /data/yb/q-$t/wq; done) > $O/rowerr2-edy.jsonl
  python3 $X/f1fam.py $O/rowerr2-edy.jsonl $W10,/data/yb --json $O/f1fam.json | tee -a $O/result.txt
else
  log "step 1 waits for w10-hout-attrib (no DONE in $W10/result.txt) or is done"
fi
log W08-HPRICE DONE
