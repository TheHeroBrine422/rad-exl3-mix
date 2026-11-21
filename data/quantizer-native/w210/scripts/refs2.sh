#!/bin/bash
# w210 box2: the reference runs, trimmed order (replaces refs.sh + refsw.sh after their first runs; finished tags are skipped). refs2.sh CARD
C=$1; cd /data/q27/w210
OFF="--set quant.block_order.enable=false --set post.c9.enable=false"
NOH="--set inputs.hout.enable=false"; NOCD="--set quant.tile_cd.enable=false"
ALL=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj
PLAIN="--set quant.refit=true --set quant.scaled_hessian=false --set quant.sigma_reg_in=0.025 --set quant.drift=\"interp\" --set quant.scale_clamp=false"
./ref.sh $C release-K4 4 $ALL $OFF
./ref.sh $C cd-K4 4 self_attn.q_proj,self_attn.k_proj $OFF $NOH --set quant.refit=true --set 'quant.tile_cd.families=["attn_qg","attn_k"]'
./ref.sh $C hout-K4 4 self_attn.q_proj,self_attn.k_proj,self_attn.o_proj,mlp.gate_proj $OFF $NOCD
./ref.sh $C houtns-K4 4 self_attn.k_proj,self_attn.o_proj $OFF $NOCD --set quant.scaled_hessian=false --set quant.sigma_reg_in=0.025
export WARM=1   # the timing runs: layer 7's k_proj first takes the process's start-up
./ref.sh $C ladderw-K4 4 $ALL $OFF $NOH $NOCD --set quant.refit=true
./ref.sh $C releasew-K4 4 $ALL $OFF
./ref.sh $C plainw-K4 4 $ALL $OFF $NOH $NOCD $PLAIN
./ref.sh $C ladderw-K2 2 $ALL $OFF $NOH $NOCD --set quant.refit=true
./ref.sh $C releasew-K2 2 $ALL $OFF
./ref.sh $C plainw-K2 2 $ALL $OFF $NOH $NOCD $PLAIN
unset WARM
./ref.sh $C clamp-K0.125 0.125 self_attn.k_proj,self_attn.v_proj,self_attn.o_proj $OFF $NOH $NOCD --set quant.refit=true
./ref.sh $C noclamp-K0.125 0.125 self_attn.k_proj,self_attn.v_proj,self_attn.o_proj $OFF $NOH $NOCD --set quant.refit=true --set quant.scale_clamp=false
echo DONE > out/ref/done2
