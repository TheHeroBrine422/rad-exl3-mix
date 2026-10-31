#!/bin/bash
# convemb.sh NAME SRC BITS [REUSE.rad] -- /root/models/final/q38-27b-NAME.rad: the ladder recipe from the exl3 export SRC,
# token_embd a BITS-bit Lloyd codebook under fwht128 (codebook from /root/levels.txt; served by embed_tq)
exec 9>/root/heavy.lock; flock 9   # one heavy job at a time: the GPU box's RAM holds a conversion OR a radiance, not both
N=$1 SRC=$2 B=$3 RU=$4
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance RADIANCE_HOST_DEVICES=1
mkdir -p /root/models/final; cd /workspace
T=$(grep "^$B " /root/levels.txt | cut -d" " -f2)
[ -n "$T" ] || { echo "$N: no $B-bit codebook"; exit 1; }
R=/root/models/final/$N.recipe
: > $R
for w in blk.*.attn_qg blk.*.attn_k blk.*.attn_v blk.*.attn_output blk.*.ffn_gate_up blk.*.ffn_down \
         blk.*.ssm_qkv blk.*.ssm_z blk.*.ssm_out output; do
  echo "$w.weight exl3 src=$SRC codebook=mul1" >> $R
done
echo "token_embd.weight rtn table=$T group=128 scale=bf16 transform=fwht128 rule=search" >> $R
O=/root/models/final/q38-27b-$N.rad
s=$(date +%s)
nice rad-convert models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
  ${RU:+--reuse $RU} -o $O.part > /root/models/final/$N.log 2>&1 && mv $O.part $O
rc=$?
echo "$N rc=$rc $(( $(date +%s) - s ))s $( [ -f $O ] && python3 -c "import os;print(round(os.path.getsize('$O')*8/26895998464,3),'real bpw')")" >> /root/models/final/conv.out
