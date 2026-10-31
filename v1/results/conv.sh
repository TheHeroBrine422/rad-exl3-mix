#!/bin/bash
# conv.sh -- on the card, after the bf16 checkpoint lands: (1) 2.00 with an fp8 token table (radiance's rtn,
# embed_lookup_q), reusing every exl3 weight of the existing 2.00 container; (2) the self-calibrated 4.00 (SC_4.00bpw_H5)
while ! grep -q DLDONE /root/dlbf16.out 2>/dev/null; do sleep 30; done
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance RADIANCE_HOST_DEVICES=1
cd /workspace
ladder() {  # SRC -> the ladder recipe's lines
  for w in blk.*.attn_qg blk.*.attn_k blk.*.attn_v blk.*.attn_output blk.*.ffn_gate_up blk.*.ffn_down \
           blk.*.ssm_qkv blk.*.ssm_z blk.*.ssm_out output; do
    echo "$w.weight exl3 src=$1 codebook=mul1"
  done
}
{ ladder /workspace/references/td-SC_2.00bpw_H3; echo "token_embd.weight rtn codes=fp8_e4m3 block=*x* scale=bf16"; } > /root/r-2.00-fp8.recipe
nice rad-convert models/qwen38-27b-bf16 --recipe /root/r-2.00-fp8.recipe --set quantization_config.quant_method=exl3 \
  --reuse /root/models/q38-27b-exl3-2.00bpw.rad -o /root/emb/2.00-fp8real.rad > /root/conv-2.00-fp8.log 2>&1
echo "fp8real rc=$?" >> /root/conv.out
ladder /workspace/references/td-SC_4.00bpw_H5 > /root/r-4.00sc.recipe
nice rad-convert models/qwen38-27b-bf16 --recipe /root/r-4.00sc.recipe --set quantization_config.quant_method=exl3 \
  -o /root/models/q38-27b-exl3-4.00scbpw.rad.part > /root/conv-4.00sc.log 2>&1 && mv /root/models/q38-27b-exl3-4.00scbpw.rad.part /root/models/q38-27b-exl3-4.00scbpw.rad
echo "4.00sc rc=$?" >> /root/conv.out
