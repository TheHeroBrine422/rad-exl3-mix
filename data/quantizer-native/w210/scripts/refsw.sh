#!/bin/bash
# w210 box2: the layer-timing references again with a warm-up tensor first (layer 7's k_proj takes the process's start-up). refsw.sh CARD
C=$1; cd /data/q27/w210; export WARM=1
OFF="--set quant.block_order.enable=false --set post.c9.enable=false"
NOH="--set inputs.hout.enable=false"; NOCD="--set quant.tile_cd.enable=false"
ALL=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj
for K in 4 2; do
  ./ref.sh $C ladderw-K$K $K $ALL $OFF $NOH $NOCD --set quant.refit=true
  ./ref.sh $C releasew-K$K $K $ALL $OFF
  ./ref.sh $C plainw-K$K $K $ALL $OFF $NOH $NOCD --set quant.refit=true --set quant.scaled_hessian=false --set quant.sigma_reg_in=0.025 --set 'quant.drift="interp"' --set quant.scale_clamp=false
done
echo DONE > out/ref/donew
