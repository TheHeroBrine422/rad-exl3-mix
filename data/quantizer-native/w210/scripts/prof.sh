#!/bin/bash
# w210 box2: layer 3 (7 linears) through exl3v3_tool tensor. usage: prof.sh TAG BUILD_DIR CARD "K list" [extra tool args]
TAG=$1; B=$2; CARD=$3; KS=$4; shift 4
cd /data/q27/w210; . ./env.sh; export ROCR_VISIBLE_DEVICES=$CARD OMP_NUM_THREADS=8
CK=/data/models/qwen38-27b-bf16; HS=/data/q27/challenge/hessians/calib; P=model.language_model.layers.3
mkdir -p out/$TAG rec/$TAG
for spec in self_attn.q_proj:input self_attn.k_proj:input self_attn.v_proj:input self_attn.o_proj:self_attn.o_proj.input \
            mlp.gate_proj:post_attention.input mlp.up_proj:post_attention.input mlp.down_proj:mlp.down_proj.input; do
  key=${spec%:*}; hn=${spec#*:}; o=out/$TAG/$key.jsonl
  [ -s $o ] && continue
  flock /data/q27/locks/card$CARD.lock $B/exl3v3_tool tensor --key $P.$key --hname $P.$hn --K $KS --ckpt $CK --hess $HS --out rec/$TAG/$key "$@" > $o.part 2>> out/$TAG/err && mv $o.part $o
done
rm -rf rec/$TAG
echo DONE > out/$TAG/done
