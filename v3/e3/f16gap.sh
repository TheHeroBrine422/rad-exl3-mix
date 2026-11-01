#!/bin/bash
# f16gap.sh -- is exllamav3's GPU trellis search worse because of its fp16 costs, and does that survive on real tensors?
# 1. Gaussian tiles, CPU search exact vs fp16-emulated (viterbi_set_f16): should reproduce fracsmoke's GPU gap
#    (cpu/gpu 0.936 at K1 ... 0.967 at K5) if fp16 costs are the cause.
# 2. 16 real tensors with block LDLQ, 128 rows, K2/3/4/6, exact vs fp16 (trellis.py E3_F16). Out: data/f16/.
cd "$(dirname "$0")"; mkdir -p data/f16
export E3_CKPT=${E3_CKPT:-/data/models/qwen38-27b-bf16} E3_HESS=${E3_HESS:-/data/challenge/hessians/calib} OMP_NUM_THREADS=4
OMP_NUM_THREADS=16 python3 - > data/f16/gauss.txt <<'PY'
import ctypes, numpy as np, trellis as T
x = np.random.default_rng(7).standard_normal((2048, 256)).astype(np.float32)
cb = np.ascontiguousarray(T.codebook("mul1"), dtype=np.float32)
for K in (1, 2, 2.5, 3, 4, 5, 6):
    T.LIB.viterbi_set_f16(0); e0 = float(((T.vit(x, cb, K) - x) ** 2).mean())
    T.LIB.viterbi_set_f16(1); e1 = float(((T.vit(x, cb, K) - x) ** 2).mean())
    print(f"K{K}: exact {e0:.4e}  fp16 {e1:.4e}  exact/fp16 {e0 / e1:.3f}", flush=True)
PY
echo "gaussian done" >&2
one() { [ -s data/f16/$1-$2-f$3.jsonl ] && return
  E3_F16=$3 nice python3 trellis.py $1 $2 mul1 2,3,4,6 128 data/f16/$1-$2-f$3.jsonl > data/f16/$1-$2-f$3.log 2>&1; echo "$(date -u +%H:%M) rc=$? $*"; }
export -f one
for t in "0 ssm_qkv" "0 ffn_gate_up" "0 ffn_down" "10 ssm_out" "10 ssm_z" "10 ssm_qkv" "10 ffn_down" "11 attn_qg" \
         "11 attn_k" "11 attn_v" "11 attn_output" "11 ffn_gate_up" "43 attn_output" "43 ffn_down" "63 attn_qg" "63 ffn_down"; do
  for f in 0 1; do echo "$t $f"; done; done | xargs -P ${JOBS:-11} -L 1 bash -c 'one $0 $1 $2'
echo F16GAP DONE
