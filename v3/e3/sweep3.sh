#!/bin/bash
# sweep3.sh -- sweep2's grid on more depths: layers 0, 62 (GDN) and 31, 63 (attention), every linear.
#   e3.py (1024 sampled rows): gptq_rot, gptq_g128i8, lloyd_rot at 2,3,4,5,6,8 bits      -> sweep3.jsonl
#   trellis (256 rows): mul1, mul1_i8, gauss_i8, int4grid at K 2..6                     -> tsweep3.jsonl
# Resumable (skipdone.py); ffn_down in a second phase at lower concurrency (~7 GB a process). Env: P, LOG, E3_CKPT, E3_HESS.
: ${RXM_LOGS:=/workspace/logs}; export RXM_LOGS   # paths (README: Paths)
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOG=${LOG:-${RXM_LOGS}}
GDN="ssm_qkv ssm_z ssm_out ffn_gate_up"; ATT="attn_qg attn_k attn_v attn_output ffn_gate_up"
jobs() {   # $1: down / notdown
  for lt in $(for l in 0 62; do for t in $GDN ffn_down; do echo $l:$t; done; done; for l in 31 63; do for t in $ATT ffn_down; do echo $l:$t; done; done); do
    l=${lt%%:*}; t=${lt#*:}
    [ "$1" = down ] && [ $t != ffn_down ] && continue; [ "$1" = notdown ] && [ $t = ffn_down ] && continue
    for K in 2 3 4 5 6; do for cb in mul1 mul1_i8 gauss_i8 int4grid; do echo "trellis.py $l $t $cb $K 256 tsweep3.jsonl"; done; done
    for m in gptq_rot gptq_g128i8 lloyd_rot; do echo "e3.py $l $t $m 2,3,4,5,6,8 sweep3.jsonl"; done
  done
}
jobs notdown | python3 skipdone.py sweep3.jsonl tsweep3.jsonl | xargs -P ${P:-16} -L 1 sh -c 'python3 -W ignore "$@" > /dev/null 2>> $LOG/e3-sweep3.err' sh
jobs down    | python3 skipdone.py sweep3.jsonl tsweep3.jsonl | xargs -P ${PD:-5} -L 1 sh -c 'python3 -W ignore "$@" > /dev/null 2>> $LOG/e3-sweep3.err' sh
echo SWEEP3DONE >> $LOG/e3-sweep3.err
