#!/bin/bash
# w210 box2: the card pipeline (build) against w209's host pipeline (build-base): layer 3's 7 tensors at K2 / K4, records and scales byte for byte
cd /data/q27/w210; mkdir -p out/ident idrec; : > out/ident/ident.txt.part
for spec in self_attn.q_proj:input self_attn.k_proj:input self_attn.v_proj:input self_attn.o_proj:self_attn.o_proj.input \
            mlp.gate_proj:post_attention.input mlp.up_proj:post_attention.input mlp.down_proj:mlp.down_proj.input; do
  key=${spec%:*}; hn=${spec#*:}
  ./t1.sh build $1 $key $hn 2,4 --out /data/q27/w210/idrec/new > /dev/null
  ./t1.sh build-base $1 $key $hn 2,4 --out /data/q27/w210/idrec/base > /dev/null
  for K in 2 4; do for p in record.u32 suh.f16 svh.f16; do
    cmp -s idrec/new.k$K.$p idrec/base.k$K.$p && r=identical || r=DIFFERENT
    echo "$key K$K $p $r $(stat -c %s idrec/new.k$K.$p) bytes" >> out/ident/ident.txt.part
  done; done
done
rm -rf idrec; mv out/ident/ident.txt.part out/ident/ident.txt
