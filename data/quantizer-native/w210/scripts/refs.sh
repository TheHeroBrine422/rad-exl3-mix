#!/bin/bash
# w210 box2: the reference runs (rxq = the release pipeline) for the extras and the layer timings. refs.sh CARD
C=$1; cd /data/q27/w210
OFF="--set quant.block_order.enable=false --set post.c9.enable=false"
NOH="--set inputs.hout.enable=false"; NOCD="--set quant.tile_cd.enable=false"
ALL=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj
# ladder settings (no H_out, no CD; refit as the fork's), every tensor: timing + proxy
for K in 4 2; do ./ref.sh $C ladder-K$K $K $ALL $OFF $NOH $NOCD --set quant.refit=true; done
# the release's settings minus block order / C9: H_out (blocks, group), hout refit, CD on q, clamp
for K in 4 2; do ./ref.sh $C release-K$K $K $ALL $OFF; done
# w209's plain settings (ourq.py --codebook mad --qtf2 defaults): no scaled Hessian, sigma 0.025 everywhere, exllamav3's drift
for K in 4 2; do ./ref.sh $C plain-K$K $K $ALL $OFF $NOH $NOCD --set quant.refit=true --set quant.scaled_hessian=false --set quant.sigma_reg_in=0.025 --set 'quant.drift="interp"' --set quant.scale_clamp=false; done
# tile CD without H_out on q and k
./ref.sh $C cd-K4 4 self_attn.q_proj,self_attn.k_proj $OFF $NOH --set quant.refit=true --set 'quant.tile_cd.families=["attn_qg","attn_k"]'
# H_out without CD
./ref.sh $C hout-K4 4 self_attn.q_proj,self_attn.k_proj,self_attn.o_proj,mlp.gate_proj $OFF $NOCD
# H_out without the scaled Hessian (the two-sided proxy then has one definition on both sides)
./ref.sh $C houtns-K4 4 self_attn.k_proj,self_attn.o_proj $OFF $NOCD --set quant.scaled_hessian=false --set quant.sigma_reg_in=0.025
# the clamp's level: K0.125 (sigma 0.2 on both groups there), no H_out, no CD
./ref.sh $C clamp-K0.125 0.125 self_attn.k_proj,self_attn.v_proj,self_attn.o_proj $OFF $NOH $NOCD --set quant.refit=true
./ref.sh $C noclamp-K0.125 0.125 self_attn.k_proj,self_attn.v_proj,self_attn.o_proj $OFF $NOH $NOCD --set quant.refit=true --set quant.scale_clamp=false
echo DONE > out/ref/done
