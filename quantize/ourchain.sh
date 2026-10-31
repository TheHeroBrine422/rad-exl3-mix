#!/bin/bash
# ourchain.sh -- card 0's queue: our EXL3 quantization on the challenge calib Hessians.
#  (a head peaks at ~22 GB of host RAM, so card 1's evals pause while one quantizes)
#  1. heads K=8..3 -> $RXM_OURS/hd-K$K.rad (SC 4.00 with our head), queued for a head-swap eval on the calibration
#     worker's queue; deleted once scored.
#  2. full models K=8,7,6,5,4,3,2 (+head) -> $RXM_OURS/ours-K$K.rad (every linear ours, embedding SC 4.00's),
#     proxy-scored (hproxy -> $RXM_WORK/hproxy-ours.jsonl), export deleted. Stops when the disk would not hold the next.
# Log: $RXM_WORK/ourchain.log
: ${RXM_WORK:=/root} ${RXM_CKPT:=/data/workspace/models/qwen38-27b-bf16} ${RXM_VENV:=/data/venv-exl3} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_CKPT RXM_VENV RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}; . ${RXM_VENV}/bin/activate
export ROCR_VISIBLE_DEVICES=0
Q=${RXM_WORK}/evq/queue-cal.txt
ALL='blk.*.attn_qg.weight blk.*.attn_k.weight blk.*.attn_v.weight blk.*.attn_output.weight blk.*.ffn_gate_up.weight blk.*.ffn_down.weight blk.*.ssm_qkv.weight blk.*.ssm_z.weight blk.*.ssm_out.weight output.weight'
log() { echo "$(date +%H:%M) $*" >> ${RXM_WORK}/ourchain.log; }
q() { python -I ourq/ourq.py "$@" 2>&1 | grep -v -e Warning -e "cpu = _conv"; return ${PIPESTATUS[0]}; }
# a head peaks at ~22 GB of host RAM: an eval's pinned memory does not leave that, so card 1's evals pause meanwhile
qhead() { ( exec 9>${RXM_WORK}/card1.lock; flock 9; q "$@" ); }
for K in 8 7 6 5 4 3; do
  [ -f evq/scores/head-sc4.00-e4-ours-K$K.f32 ] && continue
  qhead -K $K -o ${RXM_OURS}/K$K --only-head >> ${RXM_WORK}/ourq-K$K.log || { log "head K$K quantize FAIL"; continue; }
  ./ourq/ourconv.sh ${RXM_OURS}/hd-K$K.rad sc4.00-e4 ${RXM_OURS}/K$K output.weight >> ${RXM_WORK}/ourchain.log 2>&1 || { log "head K$K conv FAIL"; continue; }
  echo "head-sc4.00-e4-ours-K$K ${RXM_OURS}/hd-K$K.rad -" >> $Q
  ( until [ -f evq/scores/head-sc4.00-e4-ours-K$K.f32 ] || grep -q "\"tag\": \"head-sc4.00-e4-ours-K$K\", .*fail" evq/results.jsonl; do sleep 60; done
    rm -f ${RXM_OURS}/hd-K$K.rad ) &
  log "head K$K queued"
done
for K in 8 7 6 5 4 3 2; do
  O=${RXM_OURS}/ours-K$K.rad
  [ -f $O ] && continue
  need=$(( 27 * K / 8 * 2 + 10 ))   # GB: export + container + margin
  free=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
  [ $free -lt $need ] && { log "K$K: $free GB free < $need needed, stopping"; break; }
  s=$(date +%s)
  q -K $K -o ${RXM_OURS}/K$K >> ${RXM_WORK}/ourq-K$K.log && qhead -K $K -o ${RXM_OURS}/K$K --only-head >> ${RXM_WORK}/ourq-K$K.log \
    || { log "K$K quantize FAIL"; continue; }
  log "K$K quantized $(( ($(date +%s) - s) / 60 )) min"
  ./ourq/ourconv.sh $O sc4.00-e4 ${RXM_OURS}/K$K $ALL >> ${RXM_WORK}/ourchain.log 2>&1 || { log "K$K conv FAIL"; continue; }
  OMP_NUM_THREADS=8 ./hproxy/hproxy ${RXM_CKPT} challenge/hessians/calib hproxy-ours.jsonl ours-K$K=$O > hproxy-ours-K$K.log 2>&1 \
    && rm -rf ${RXM_OURS}/K$K && log "K$K container + proxy done, export removed" || log "K$K hproxy FAIL (export kept)"
done
wait
log OURCHAINDONE
