#!/bin/bash
# w210 box2: the final build (build2 = the last commit) against w209's host pipeline on two tensors, and --scale-sample 4 on real tensors. final.sh CARD
C=$1; cd /data/q27/w210; mkdir -p out/final frec; O=out/final
for spec in self_attn.k_proj:input mlp.gate_proj:post_attention.input; do
  key=${spec%:*}; hn=${spec#*:}
  ./t1.sh build2 $C $key $hn 2,4 --exl3-drift --out /data/q27/w210/frec/new > /dev/null
  ./t1.sh build-base $C $key $hn 2,4 --out /data/q27/w210/frec/base > /dev/null
  for K in 2 4; do for p in record.u32 suh.f16 svh.f16; do
    cmp -s frec/new.k$K.$p frec/base.k$K.$p && r=identical || r=DIFFERENT
    echo "$key K$K $p $r" >> $O/ident2.txt.part
  done; done
done
mv $O/ident2.txt.part $O/ident2.txt; rm -rf frec
for spec in self_attn.q_proj:input:0.005 self_attn.k_proj:input:0.005 self_attn.v_proj:input:0.005 self_attn.o_proj:self_attn.o_proj.input:0.025 mlp.gate_proj:post_attention.input:0.025; do
  IFS=: read key hn sg <<< "$spec"
  [ -s $O/ss4.$key.jsonl ] || { ./t1.sh build2 $C $key $hn 2,4 --scaled-hessian --sigma $sg --scale-sample 4 --warm > $O/ss4.$key.jsonl.part && mv $O/ss4.$key.jsonl.part $O/ss4.$key.jsonl; }
done
echo DONE > $O/done
