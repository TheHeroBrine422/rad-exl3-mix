#!/bin/bash
# gpucheck.sh UNIT -- w50's 4070 checks of ourq --tile-cd (README.md here), 27B attn_qg (q_proj) of layers 3 / 19 / 43
# (settle.py's item-3 tensors), the ladder's settings (qtf2, i8, scaled Hessian, sigma 0.025 / input 0.005).
#   check   arms orig (today's ourq.py), off (patched, no --tile-cd), cd3 (patched, --tile-cd 3) at K2,3,4 with
#           --save-wq; then: orig vs off byte-identical shards (codes + code stamp), stamps, cd3 / off ratios
#   time    arms toff / tcd3 (patched, without / with --tile-cd 3) at K2,2.5,3,4 without --save-wq: per-tensor time;
#           first the arm cd3f64 (layers 3 / 19, K2, --save-wq; TILECD_DTYPE=float64: is layer 3's gap vs the CPU
#           proxy fp32 precision?)
#   fam L[,L] [F,F]  layer L whole (every family) at K2.5 without / with --tile-cd 3 on every family (arms foffL / fcdL): is
#           attn_qg's gain on the card larger than the other families'? (proxy_err ratios; per-tensor times)
# One unit = one hold of /data/yb/gpu.lock (< 15 min, timeout per run). VRAM capped at VRAM_CAP_GB (7.2) by
# capq.py. Scratch /tmp/w50 (code copy at a fixed path, so the stamps compare); outputs /tmp/w50/out/<arm>.
set -u
U=${1:?check|time}; R=/workspace/rad-exl3-mix; X=/tmp/w50; C=$X/code; O=$X/out
SRC=/data/src/exllamav3/exllamav3/exllamav3_ext W=/data/p27b
mkdir -p $C/quantize $O $X/run; ln -sfn $R/v3 $C/v3
L="model.language_model.layers"; printf "$L.3.self_attn.q_proj\n$L.19.self_attn.q_proj\n$L.43.self_attn.q_proj\n" > $X/only.txt
printf "$L.3.self_attn.q_proj\n$L.19.self_attn.q_proj\n" > $X/only2.txt
run() { arm=$1 ks=$2; shift 2; only=${ONLY:-$X/only.txt}; last=${ks##*,}
  [ -f $O/$arm/K$last/q-L$(tail -1 $only | cut -d. -f4 | xargs printf %02d).safetensors ] && { echo "skip $arm"; return 0; }
  s=$(date +%s)
  ( cd $X/run && OURQ_REFIT_CPU_FALLBACK=1 PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=/workspace/models/qwen38-27b-bf16 \
    RXM_TEMPLATE=$W/template.json TORCH_EXTENSIONS_DIR=/data/torch_ext TILECD_DTYPE=${TILECD_DTYPE:-float32} VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 400 \
    python3 $R/v3/e3/tilecd/capq.py $C/quantize/ourq.py -K $ks -o "$O/$arm/K{K}" --device 0 --layers 0-63 --only $only \
    --codebook i8 --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 "$@" ) > $O/$arm.log 2>&1
  echo "$arm rc=$? $(( $(date +%s) - s ))s"; }
exec 9>/data/yb/gpu.lock; t0=$(date +%s); flock 9; t1=$(date +%s)
echo "lock $(date -u +%T) after $(( t1 - t0 ))s"
ALLF=attn_qg,attn_k,attn_v,attn_output,ssm_qkv,ssm_z,ssm_out,ffn_gate_up,ffn_down
runfam() { arm=$1 l=$2; shift 2
  [ -f $O/$arm/K2.5/q-L$(printf %02d $l).safetensors ] && { echo "skip $arm"; return 0; }
  s=$(date +%s)
  ( cd $X/run && OURQ_REFIT_CPU_FALLBACK=1 PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=/workspace/models/qwen38-27b-bf16 \
    RXM_TEMPLATE=$W/template.json TORCH_EXTENSIONS_DIR=/data/torch_ext VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 480 \
    python3 $R/v3/e3/tilecd/capq.py $C/quantize/ourq.py -K 2.5 -o "$O/$arm/K{K}" --device 0 --layers $l-$l \
    --codebook i8 --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 "$@" ) > $O/$arm.log 2>&1
  echo "$arm rc=$? $(( $(date +%s) - s ))s"; }
case $U in
check)
  cp $R/quantize/ourq.py $C/quantize/ourq.py; run orig 2,3,4 --save-wq
  patch -s -o $C/quantize/ourq.py $R/quantize/ourq.py $R/v3/e3/tilecd/ourq-tilecd.patch
  run off 2,3,4 --save-wq
  run cd3 2,3,4 --save-wq --tile-cd 3 ;;
time)
  patch -s -o $C/quantize/ourq.py $R/quantize/ourq.py $R/v3/e3/tilecd/ourq-tilecd.patch
  TILECD_DTYPE=float64 ONLY=$X/only2.txt run cd3f64 2 --save-wq --tile-cd 3
  run toff 2,2.5,3,4
  run tcd3 2,2.5,3,4 --tile-cd 3 ;;
fam)   # fam L[,L..] [FAMILIES]: ffn_down CD needs > 7.2 GB at 27B (OOM on the 4070's cap): leave it out there
  patch -s -o $C/quantize/ourq.py $R/quantize/ourq.py $R/v3/e3/tilecd/ourq-tilecd.patch
  for l in ${2//,/ }; do runfam foff$l $l; runfam fcd$l $l --tile-cd 3 --tile-cd-families ${3:-$ALLF}; done ;;
esac
flock -u 9; echo "released $(date -u +%T), held $(( $(date +%s) - t1 ))s"
echo "GPUCHECK $U DONE"
