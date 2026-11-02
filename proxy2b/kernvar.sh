#!/bin/bash
# kernvar.sh -- which GDN kernel path agrees with the torch reference: a 2-document golden per path, cross-scored.
# T = pure torch (P2B_BLOCK=causal_conv1d,fla), F = fla delta rule + torch conv, FC = fla + causal_conv1d.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-2b O=/data/kernvar; mkdir -p $O
declare -A B=([T]="causal_conv1d,fla" [F]="causal_conv1d" [FC]="")
for v in T F FC; do [ -f $O/$v/kld.json ] || P2B_BLOCK=${B[$v]} python3 $P/mkgold.py $M /data/challenge/calib-meta $O/$v 0:2048 1:2048 > $O/mk-$v.log 2>&1; done
for pair in "FC T" "F T" "FC F" "T T"; do set -- $pair
  echo "$1 vs golden $2: $(P2B_BLOCK=${B[$1]} P2B_ANY_KERNELS=1 python3 $P/evalq.py $M $O/$2 2>/dev/null | tail -1)"; done
