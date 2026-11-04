#!/bin/bash
# hconv.sh -- quantizer-level convergence of the 27B H_out collection (4070, one GPU job at a time; README.md).
# Layers 10 (GDN), 39 (full attention), 50 (GDN) at K2.5 with the ladder's settings, quantized with: no H_out ("none")
# and the even-block Σdy² diagonal up to each snapshot N (train-N, from mkhalf.py); scored on the odd-block diagonal
# (test, independent) by hscore.py. Idempotent: new snapshots -> new arms; rerun after each sync.
R=/workspace/rad-exl3-mix V=$R/v3/yaqa/hconv O=/data/yb/hconv M=/data/models/qwen38-27b-bf16 HS=/data/challenge/dl/hessians/calib
LAYERS=${LAYERS:-"10 39 50"} K=${K:-2.5}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
while pgrep -f "quantize/[o]urq.py|proxy2b/[e]valq.py|yaqa/[c]ollect.py" > /dev/null; do sleep 60; done
python3 -I $V/mkhalf.py $O/snaps $O/half dy | tee -a $O/result.txt
W=/data/p27b; mkdir -p $W/challenge/hessians; ln -sfn $HS $W/challenge/hessians/calib
python3 -c "import json; c=json.load(open('$M/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$W/template.json', 'w'))"
SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
q() { arm=$1; shift; for L in $LAYERS; do
  [ -f $O/q-$arm/q-L$(printf %02d $L).safetensors ] && continue
  ( mkdir -p /tmp/hconv-run && cd /tmp/hconv-run && OURQ_REFIT_CPU_FALLBACK=1 PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=$M RXM_TEMPLATE=$W/template.json \
    TORCH_EXTENSIONS_DIR=/data/torch_ext python3 $R/quantize/ourq.py -K $K -o $O/q-$arm --device 0 --layers $L-$L --codebook i8 --qtf2 \
    --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq "$@" ) >> $O/q-$arm.log 2>&1 || log "QUANT FAIL $arm L$L"
done; }
q none
for d in $(ls -d $O/half/train-* | sort -V); do n=${d##*-}; q dy$n --hout $d --sigma-reg-out 0.01; done
ARMS="none=$O/q-none/wq"; for d in $(ls -d $O/q-dy* | sort -V); do ARMS="$ARMS ${d##*/q-}=$d/wq"; done
python3 -I $V/hscore.py $M $HS $O/half/test $ARMS > $O/score.jsonl 2> $O/score.err
python3 - $O/score.jsonl <<'PY' | tee -a $O/result.txt
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]; base = {r["key"]: r for r in R if r["arm"] == "none"}
by = collections.defaultdict(list)
for r in R: by[r["arm"]].append(r)
def n(a): return -1 if a == "none" else int(a[2:])
print(f"{'arm':>7} {'tensors':>7} {'E_out / none (param-wtd)':>25} {'E_in / none':>12}")
for a in sorted(by, key=n):
    rs = [r for r in by[a] if r["key"] in base]; w = sum(r["params"] for r in rs)
    eo = sum(r["params"] * r["E_out"] / base[r["key"]]["E_out"] for r in rs) / w
    ei = sum(r["params"] * r["E_in"] / base[r["key"]]["E_in"] for r in rs) / w
    print(f"{a:>7} {len(rs):7d} {eo:25.4f} {ei:12.4f}")
PY
log "HCONV DONE"
