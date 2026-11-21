#!/bin/bash
# w210 box2: the scale search's g at low rates, new vs rxq (k_proj, v_proj; sigma as the release: 0.2 at K0.125, 0.05 below K1, 0.005 at K1). sub1.sh CARD
C=$1; cd /data/q27/w210; mkdir -p out/sub1
OFF="--set quant.block_order.enable=false --set post.c9.enable=false --set inputs.hout.enable=false --set quant.tile_cd.enable=false --set quant.refit=true"
for ks in 0.25:0.05 0.5:0.05 1:0.005; do
  K=${ks%:*}; sg=${ks#*:}
  ./ref.sh $C sub1-K$K $K self_attn.k_proj,self_attn.v_proj $OFF
  for t in k v; do [ -s out/sub1/new-$t-K$K.jsonl ] || { ./t1.sh build2 $C self_attn.${t}_proj input $K --scaled-hessian --sigma $sg > out/sub1/new-$t-K$K.jsonl.part && mv out/sub1/new-$t-K$K.jsonl.part out/sub1/new-$t-K$K.jsonl; }; done
done
for t in k v; do [ -s out/sub1/new-$t-K0.125-f32.jsonl ] || { ./t1.sh build2 $C self_attn.${t}_proj input 0.125 --scaled-hessian --sigma 0.2 --f32 > out/sub1/new-$t-K0.125-f32.jsonl.part && mv out/sub1/new-$t-K0.125-f32.jsonl.part out/sub1/new-$t-K0.125-f32.jsonl; }; done
for t in k v; do [ -s out/sub1/new-$t-K0.125-seed7.jsonl ] || { ./t1.sh build2 $C self_attn.${t}_proj input 0.125 --scaled-hessian --sigma 0.2 --seed 7 > out/sub1/new-$t-K0.125-seed7.jsonl.part && mv out/sub1/new-$t-K0.125-seed7.jsonl.part out/sub1/new-$t-K0.125-seed7.jsonl; }; done
echo DONE > out/sub1/done
