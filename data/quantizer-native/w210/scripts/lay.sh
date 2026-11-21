#!/bin/bash
# w210 box2: layer 3 through exl3v3_tool with a settings set. lay.sh TAG CARD "K list" MODE   (MODE: ladder | release)
#   ladder  = scaled Hessian, sigma_in 0.005 (attn q / k / v), 0.025 elsewhere, scale clamp
#   release = ladder + the output Hessian (two-sided LDLQ, hout-weighted refit) on every tensor + tile CD 3 on q_proj
TAG=$1; CARD=$2; KS=$3; MODE=$4
cd /data/q27/w210; mkdir -p out/$TAG
for spec in self_attn.q_proj:input:0.005 self_attn.k_proj:input:0.005 self_attn.v_proj:input:0.005 self_attn.o_proj:self_attn.o_proj.input:0.025 \
            mlp.gate_proj:post_attention.input:0.025 mlp.up_proj:post_attention.input:0.025 mlp.down_proj:mlp.down_proj.input:0.025; do
  IFS=: read key hn sg <<< "$spec"; o=out/$TAG/$key.jsonl
  [ -s $o ] && continue
  X="--scaled-hessian --sigma $sg --warm"
  [ $MODE = release ] && X="$X --hout /data/q27/w210/hout/bn1920"
  [ $MODE = release ] && [ $key = self_attn.q_proj ] && X="$X --tile-cd 3"
  ./t1.sh build $CARD $key $hn $KS $X > $o.part 2>> out/$TAG/err && mv $o.part $o
done
echo DONE > out/$TAG/done
