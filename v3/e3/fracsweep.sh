#!/bin/bash
# fracsweep.sh -- fractional-K trellis (viterbi_tiles_pat) on real tensors: proxy error at quarter / half / integer K,
# same 64 rows at every K (paired), every family of layers 10 (GDN), 11 (full attention), 43 (GDN). VM, 8 threads.
cd "$(dirname "$0")"
export E3_CKPT=${E3_CKPT:-/workspace/models/qwen38-27b-bf16} E3_HESS=${E3_HESS:-/workspace/e3data/challenge/hessians/calib} OMP_NUM_THREADS=8
O=data/fracsweep.jsonl; KS=0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5
for lt in "10 ssm_out" "10 ssm_z" "10 ssm_qkv" "10 ffn_gate_up" "10 ffn_down" "11 attn_qg" "11 attn_k" "11 attn_v" "11 attn_output" \
          "43 ssm_out" "43 ssm_z" "43 ssm_qkv" "43 ffn_gate_up" "43 ffn_down"; do
  set -- $lt
  grep -q "\"layer\": $1, \"w\": \"$2\", .*\"bits\": 5.0" $O 2>/dev/null && continue
  nice python3 trellis.py $1 $2 mul1 $KS 64 $O > /dev/null || echo "FAIL $lt"
  echo "$(date -u +%H:%M) done $lt"
done
echo FRACSWEEP DONE
