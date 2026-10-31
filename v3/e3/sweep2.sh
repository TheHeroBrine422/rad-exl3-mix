#!/bin/bash
# sweep2.sh -- E3 + E3b on layers 10 (GDN) and 43 (attention), every linear, as single-thread jobs:
#   e3.py   (1024 sampled rows): rtn_rot, gptq_rot, gptq_g128i8, lloyd_rot at 2,3,4,5,6,8 bits  -> sweep2.jsonl
#   trellis (256 sampled rows):  mul1, mul1_i8, gauss_i8, int4grid at K 2..6                    -> tsweep2.jsonl
# ffn_down (17408^2 Hessian, ~7 GB a process) runs in a second phase at lower concurrency.
: ${RXM_LOGS:=/workspace/logs}; export RXM_LOGS   # paths (README: Paths)
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOG=${LOG:-${RXM_LOGS}}
T10="ssm_qkv ssm_z ssm_out ffn_gate_up"; T43="attn_qg attn_k attn_v attn_output ffn_gate_up"
jobs() {   # $1: tensors filter (down / notdown)
  for lt in $(for t in $T10 ffn_down; do echo 10:$t; done; for t in $T43 ffn_down; do echo 43:$t; done); do
    l=${lt%%:*}; t=${lt#*:}
    [ "$1" = down ] && [ $t != ffn_down ] && continue; [ "$1" = notdown ] && [ $t = ffn_down ] && continue
    for K in 2 3 4 5 6; do for cb in mul1 mul1_i8 gauss_i8 int4grid; do echo "trellis.py $l $t $cb $K 256 tsweep2.jsonl"; done; done
    for m in gptq_rot gptq_g128i8 lloyd_rot rtn_rot; do echo "e3.py $l $t $m 2,3,4,5,6,8 sweep2.jsonl"; done
  done
}
# skip what a previous run already wrote (resume after a stop, or on another machine with the jsonl copied over)
jobs notdown | python3 skipdone.py sweep2.jsonl tsweep2.jsonl | xargs -P ${P:-16} -L 1 sh -c 'python3 -W ignore "$@" > /dev/null 2>> $LOG/e3-sweep2.err' sh
jobs down    | python3 skipdone.py sweep2.jsonl tsweep2.jsonl | xargs -P ${PD:-5} -L 1 sh -c 'python3 -W ignore "$@" > /dev/null 2>> $LOG/e3-sweep2.err' sh
echo SWEEP2DONE >> $LOG/e3-sweep2.err
