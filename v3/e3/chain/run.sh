#!/bin/bash
# chain/run.sh [PDOWN] [POTHER] -- #102 on colo (README.md): one job per (tensor, K) of #57's 16 tensors (f16gap.sh),
# mul1, 128 rows, arms n = 0 (free), 1, 2, 4, 8 at K2 / 2.5 / 3 / 4; plus Gaussian tiles per K. Run from the copy
# whose parent holds e3.py, trellis.py, libviterbi.so (built there); chain/libviterbi.so from chain/viterbi.c.
# Idempotent: a job writes data/<job>.jsonl.part and renames it when done; finished and running jobs are skipped. Single-threaded
# jobs, nice 10; ffn_down jobs (~12 GB) in their own pool of PDOWN (3), the rest POTHER (3). Longest first.
cd "$(dirname "$0")"; mkdir -p data ../logs
export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
one() {  # one NAME ARGS...
  local n=$1; shift; [ -s data/$n.jsonl ] && return
  pgrep -f "trellis_chain.py .* data/$n.jsonl.part" > /dev/null && return   # running (another runner)
  rm -f data/$n.jsonl.part
  nice -n 10 python3 -W ignore trellis_chain.py "$@" data/$n.jsonl.part > ../logs/chain-$n.log 2>&1 && mv data/$n.jsonl.part data/$n.jsonl
  echo "$(date -u +%H:%M) rc=$? $n"
}
export -f one
T="0:ssm_qkv 0:ffn_gate_up 0:ffn_down 10:ssm_out 10:ssm_z 10:ssm_qkv 10:ffn_down 11:attn_qg 11:attn_k 11:attn_v 11:attn_output 11:ffn_gate_up 43:attn_output 43:ffn_down 63:attn_qg 63:ffn_down"
for K in 2 2.5 3 4; do for t in $T; do [[ $t == *ffn_down ]] && echo "t-${t/:/-}-K$K tensor ${t/:/ } mul1 $K 0,1,2,4,8 128"; done; done |
  xargs -P ${1:-3} -L 1 bash -c 'one "$@"' _ &
{ for K in 2 2.5 3 4; do for t in $T; do [[ $t == *ffn_gate_up ]] && echo "t-${t/:/-}-K$K tensor ${t/:/ } mul1 $K 0,1,2,4,8 128"; done; done
  for K in 1 2 2.5 3 4; do echo "g-K$K gauss 2048 $K 1,2,4,8,16,32"; done
  for K in 2 2.5 3 4; do for t in $T; do [[ $t != *ffn_* ]] && echo "t-${t/:/-}-K$K tensor ${t/:/ } mul1 $K 0,1,2,4,8 128"; done; done
} | xargs -P ${2:-3} -L 1 bash -c 'one "$@"' _
wait
echo CHAIN DONE
