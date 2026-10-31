#!/bin/bash
# ourchain2.sh CARD -- our EXL3 quantization on the challenge calib Hessians, one worker per card.
#  card 0: first the remaining heads K=8..3 (-> $RXM_OURS/hd-K$K.rad, SC 4.00 with our head, queued for a head-swap
#          eval on the calibration queue), then full models.
#  card 1: waits until the calib-B verification of the knapsack frontier is scored (aftercal.sh), then full models.
#  Full models: each K claimed (mkdir $RXM_WORK/ourq/claim/K$K), so the cards split them; preference card 0 8 6 4 2,
#  card 1 7 5 3, then whatever is left. -> $RXM_OURS/ours-K$K.rad (every linear ours, embedding SC 4.00's),
#  proxy-scored (hproxy -> $RXM_WORK/hproxy-ours.jsonl), export deleted.
# Locks: $RXM_WORK/cardC.lock while the card computes (the eval worker on the card waits); $RXM_WORK/ram.lock exclusive for a
# head quantization (~22 GB of the GPU box's 30 GB), shared for evals; trunk quantization takes none (peak RSS logged). Log: $RXM_WORK/ourchain.log
: ${RXM_WORK:=/root} ${RXM_CKPT:=/data/workspace/models/qwen38-27b-bf16} ${RXM_VENV:=/data/venv-exl3} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_CKPT RXM_VENV RXM_OURS   # paths (README: Paths)
C=$1
cd ${RXM_WORK}; . ${RXM_VENV}/bin/activate
export ROCR_VISIBLE_DEVICES=$C PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the dense trellis kernels for every K: bit-identical indices, K1 2.2x and K6 1.26x faster than the default pick on gfx1201
export EXL3_QT_OPTIMIZED=1
mkdir -p ${RXM_WORK}/ourq/claim
Q=${RXM_WORK}/evq/queue-cal.txt
ALL='blk.*.attn_qg.weight blk.*.attn_k.weight blk.*.attn_v.weight blk.*.attn_output.weight blk.*.ffn_gate_up.weight blk.*.ffn_down.weight blk.*.ssm_qkv.weight blk.*.ssm_z.weight blk.*.ssm_out.weight output.weight'
log() { echo "$(date +%H:%M) [c$C] $*" >> ${RXM_WORK}/ourchain.log; }
q() { python -I ourq/ourq.py "$@" 2>&1 | grep -v -e Warning -e "cpu = _conv"; return ${PIPESTATUS[0]}; }
peak() {  # run "$@", log the peak RSS of its largest python child
  "$@" & local p=$! m=0 r
  while kill -0 $p 2>/dev/null; do r=$(ps -C python -o rss= | sort -n | tail -1); [ -n "$r" ] && [ $r -gt $m ] && m=$r; sleep 5; done
  wait $p; local rc=$?; echo "peak python RSS $((m / 1024)) MiB" >> ${RXM_WORK}/ourq-peak-c$C.log; return $rc
}
# lock order everywhere: the card first, then ram.lock (evalq.sh too), so no cycle
qtrunk() { ( exec 9>${RXM_WORK}/card$C.lock; flock 9; peak q "$@" ); }
qhead()  { ( exec 9>${RXM_WORK}/card$C.lock 7>${RXM_WORK}/ram.lock; flock 9; flock 7; q "$@" ); }

if [ -n "$QORDER" ]; then          # the overnight pipeline (auto.sh): its own order, no heads loop, no gate
  ORDER="$QORDER"
elif [ $C = 0 ] && [ -z "$NOHEADS" ]; then   # NOHEADS=1: full models only (each brings its own head)
  for K in 8 7 6 5 4 3; do
    [ -f evq/scores/head-sc4.00-e4-ours-K$K.f32 ] && continue
    grep -q "^head-sc4.00-e4-ours-K$K " $Q && [ -f ${RXM_OURS}/hd-K$K.rad ] && continue   # built and queued already
    qhead -K $K -o ${RXM_OURS}/K$K --only-head >> ${RXM_WORK}/ourq-K$K.log || { log "head K$K quantize FAIL"; continue; }
    ./ourq/ourconv.sh ${RXM_OURS}/hd-K$K.rad sc4.00-e4 ${RXM_OURS}/K$K output.weight >> ${RXM_WORK}/ourchain.log 2>&1 || { log "head K$K conv FAIL"; continue; }
    grep -q "^head-sc4.00-e4-ours-K$K " $Q || echo "head-sc4.00-e4-ours-K$K ${RXM_OURS}/hd-K$K.rad -" >> $Q
    log "head K$K queued"
  done
  ORDER="8 6 4 2 7 5 3 1"
elif [ $C = 0 ]; then
  ORDER="8 6 4 2 7 5 3 1"
else
  V=evq/queue-verify.txt
  until [ -f $V ] && [ -z "$(awk '$1 ~ /^v-(knap|ref)-/ {print $1}' $V | while read t; do
          [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done)" ]; do sleep 120; done
  log "frontier verification scored; card 1 joins the quantization"
  ORDER="7 5 3 1 8 6 4 2"
fi

for K in $ORDER $ORDER; do                  # second pass: Ks that failed in the first (claim released, layers resume)
  O=${RXM_OURS}/ours-K$K.rad
  [ -f $O ] && continue
  mkdir ${RXM_WORK}/ourq/claim/K$K 2>/dev/null || continue
  s=$(date +%s)
  qtrunk -K $K -o ${RXM_OURS}/K$K >> ${RXM_WORK}/ourq-K$K.log && qhead -K $K -o ${RXM_OURS}/K$K --only-head >> ${RXM_WORK}/ourq-K$K.log \
    || { log "K$K quantize FAIL"; rm -rf ${RXM_WORK}/ourq/claim/K$K; continue; }
  log "K$K quantized $(( ($(date +%s) - s) / 60 )) min"
  ./ourq/ourconv.sh $O sc4.00-e4 ${RXM_OURS}/K$K $ALL >> ${RXM_WORK}/ourchain.log 2>&1 || { log "K$K conv FAIL"; continue; }
  OMP_NUM_THREADS=8 ./hproxy/hproxy ${RXM_CKPT} challenge/hessians/calib hproxy-ours.jsonl ours-K$K=$O > hproxy-ours-K$K.log 2>&1 \
    && rm -rf ${RXM_OURS}/K$K && log "K$K container + proxy done, export removed" || log "K$K hproxy FAIL (export kept)"
done
log "OURCHAINDONE card $C"
