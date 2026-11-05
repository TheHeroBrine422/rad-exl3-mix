#!/bin/bash
# run27.sh pilot | timing | predict -- A6 (and B6's down side) on the 27B. README "27B".
#   pilot    gpubox, BOTH cards (hdiag27.sh takes /root/card0.lock, card1.lock and ram.lock itself): the diagonal
#            collection's first NSEQ (default 256) sequences, ~15 min warm-up + ~2.1 h. Not wasted: `hdiag27.sh 1024`
#            later resumes from its hdiag-state.pt (same seed, same windows). Gives hdiag-snap256 + halves.json.
#   timing   gpubox, card 0 (card0 + shared ram lock): ldlq_2hess vs one-sided on layer 40's gate/up group at K3, from
#            the pilot's diagonal; LADDER_ARGS = ladder2.sh's ourq flags (required: the timing must be the ladder's).
#   predict  cpu1, CPU (~30 min, 36 threads): copy hdiag-snap256/, hdiag-state.pt and the ladder K3 proxy.jsonl to
#            $D first; stats.py (27B bf16 + calib Hessians) -> predict.py (A6 self + split-half cross-fit, D6, B6) ->
#            downperm.py on 6 layers (float32).
F=$(cd "$(dirname "$0")" && pwd)
case $1 in
pilot)
  bash /workspace/rad-exl3-mix/v3/yaqa/hdiag27.sh ${NSEQ:-256} ;;
timing)
  : ${LADDER_ARGS:?ladder2.sh ourq flags (codebook / qtf2 / frac-ext / damping)}; : ${RXM_WORK:=/root}
  HD=${HD:-/data/yaqa/hdiag/hdiag-snap256}; T=/data/w04/t2h; mkdir -p $T
  L=model.language_model.layers.40; printf "$L.mlp.gate_proj\n$L.mlp.up_proj\n" > $T/only.txt
  for arm in one two; do
    X=""; [ $arm = two ] && X="--hout $HD --sigma-reg-out 0.01"
    ( exec 8>$RXM_WORK/card0.lock 7>$RXM_WORK/ram.lock; flock 8; flock -s 7
      /data/venv-exl3/bin/python /workspace/rad-exl3-mix/quantize/ourq.py -K 3 -o $T/$arm --layers 40-40 --device 0 \
        --only $T/only.txt $LADDER_ARGS $X ) > $T/$arm.log 2>&1
    echo "$arm: $(python3 -c "import json; print({json.loads(l)['key'].split('.')[-1]: json.loads(l)['s'] for l in open('$T/$arm/proxy.jsonl')})")"
  done | tee $T/timing.txt ;;
predict)
  D=${D:-/data/work/w04/27b}; CK=/data/models/qwen38-27b-bf16; HIN=/data/challenge/hessians/calib
  [ -s $D/stats27.npz ] || OMP_NUM_THREADS=36 nice python3 $F/stats.py $CK $D/hdiag-snap256 $D/stats27.npz --hin $HIN \
    --state $D/hdiag-state.pt 2> $D/stats27.log || exit 1
  python3 $F/predict.py $D/pred27.json $D/stats27.npz --proxy $D/proxy-k3.jsonl --perms-out $D/perms27.npz | tee $D/pred27.txt
  python3 -c "import numpy as np; z = np.load('$D/perms27.npz'); np.savez('$D/perms27-sub.npz', **{k: z[k] for k in z.files
      if k.split('/')[0][1:] in ('0', '15', '31', '47', '62', '63') and k.endswith('/balanced')})"
  OMP_NUM_THREADS=36 DP_DTYPE=float32 DP_DRAWS=1 nice python3 $F/downperm.py $CK $HIN $D/perms27-sub.npz $D/downperm27.json \
    2>&1 | grep layer | tee $D/downperm27.txt ;;
*) echo "usage: run27.sh pilot | timing | predict"; exit 1 ;;
esac
