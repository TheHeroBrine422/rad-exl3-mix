#!/bin/bash
# w210 box2: the extras through exl3v3_tool, the runs refs.sh has references for + the ones checked against the oracle. extras.sh CARD
C=$1; cd /data/q27/w210; mkdir -p out/extras; O=out/extras
H=/data/q27/w210/hout/bn1920
run() { t=$1; shift; [ -s $O/$t.jsonl ] || { ./t1.sh build $C "$@" > $O/$t.jsonl.part 2>> $O/err && mv $O/$t.jsonl.part $O/$t.jsonl; }; }
SH="--scaled-hessian"
# tile CD without H_out (q, k), K4; and the same tensors without CD
run cd-q    self_attn.q_proj input 4 $SH --sigma 0.005 --tile-cd 3
run cd-k    self_attn.k_proj input 4 $SH --sigma 0.005 --tile-cd 3
run nocd-q  self_attn.q_proj input 4 $SH --sigma 0.005
run nocd-k  self_attn.k_proj input 4 $SH --sigma 0.005
# H_out (two-sided LDLQ, the hout-weighted refit), K4
run hout-q  self_attn.q_proj input 4 $SH --sigma 0.005 --hout $H
run hout-k  self_attn.k_proj input 4 $SH --sigma 0.005 --hout $H
run hout-o  self_attn.o_proj self_attn.o_proj.input 4 $SH --sigma 0.025 --hout $H
run hout-g  mlp.gate_proj post_attention.input 4 $SH --sigma 0.025 --hout $H
run nohout-o self_attn.o_proj self_attn.o_proj.input 4 $SH --sigma 0.025
run nohout-g mlp.gate_proj post_attention.input 4 $SH --sigma 0.025
run houtns-k self_attn.k_proj input 4 --sigma 0.025 --hout $H
run houtns-o self_attn.o_proj self_attn.o_proj.input 4 --sigma 0.025 --hout $H
# H_out + tile CD on q (the release's attn_qg)
run houtcd-q self_attn.q_proj input 4 $SH --sigma 0.005 --hout $H --tile-cd 3
# the scale clamp's level, K0.125 (sigma 0.2), with and without
for k in k v; do run clamp-$k self_attn.${k}_proj input 0.125 $SH --sigma 0.2; run noclamp-$k self_attn.${k}_proj input 0.125 $SH --sigma 0.2 --no-clamp; done
run clamp-o self_attn.o_proj self_attn.o_proj.input 0.125 $SH --sigma 0.2
run noclamp-o self_attn.o_proj self_attn.o_proj.input 0.125 $SH --sigma 0.2 --no-clamp
# R16 row order (the head's) on v_proj [1024 x 5120] and o_proj: the record decodes in the stored order
run r16-v   self_attn.v_proj input 4 $SH --sigma 0.005 --r16
run nor16-v self_attn.v_proj input 4 $SH --sigma 0.005
run r16-o   self_attn.o_proj self_attn.o_proj.input 4 $SH --sigma 0.025 --r16
# the stacked gate_up (suh.1): gate + up in one batch (one Hessian upload), the stacked record through the oracle
run stack-gu mlp.gate_proj,mlp.up_proj post_attention.input 2,4 $SH --sigma 0.025 --stack
run stack-gu-sameseed mlp.gate_proj,mlp.up_proj post_attention.input 4 --sigma 0.025 --stack --same-seed
# swj: the grid sums and fit against rxq/swj.py on the same inputs (the reference in its own shell: the fork's torch env)
if [ ! -s $O/swj.jsonl ]; then
  P=model.language_model.layers.3; mkdir -p swjtmp
  ( . ./env.sh; export ROCR_VISIBLE_DEVICES=$C OMP_NUM_THREADS=8
    flock /data/q27/locks/card$C.lock build/exl3v3_tool swj --gate $P.mlp.gate_proj --up $P.mlp.up_proj --hname $P.post_attention.input --K 2.5 \
      --scaled-hessian --ckpt /data/models/qwen38-27b-bf16 --hess /data/q27/challenge/hessians/calib --rows 2048 --dump swjtmp/s ) > out/extras/swj.jsonl.part 2>> out/extras/err
  ( . /data/q27/w209/old/env.sh; export ROCR_VISIBLE_DEVICES=$C PYTHONDONTWRITEBYTECODE=1 TORCH_EXTENSIONS_DIR=/data/q27/w210/ref/extbuild
    flock /data/q27/locks/card$C.lock python src/tests/exl3v3/ref/swj_compare.py swjtmp/s /data/models/qwen38-27b-bf16 $P.mlp.gate_proj $P.mlp.up_proj /data/q27/w210/ref/code/pipeline ) >> out/extras/swj.jsonl.part 2>> out/extras/err
  rm -rf swjtmp; [ $(wc -l < out/extras/swj.jsonl.part) -ge 2 ] && mv out/extras/swj.jsonl.part out/extras/swj.jsonl
fi
echo DONE > $O/done
