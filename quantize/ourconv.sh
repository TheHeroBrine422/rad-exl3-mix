#!/bin/bash
# ourconv.sh OUT.rad BASE_TAG SRCDIR GLOB... -- a container: the weights matching each GLOB (e.g. 'blk.0.*.weight',
# 'blk.*.ffn_down.weight', 'output.weight') imported from our export SRCDIR, the rest exactly as the final
# container BASE_TAG (its recipe, --reuse: those are copied, not re-imported). Takes the conversion lock.
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final} ${RXM_RADIANCE:=$RXM_WORK/install-card}; export RXM_WORK RXM_MODELS RXM_RADIANCE   # paths (README: Paths)
exec 9>${RXM_WORK}/conv.lock; flock 9
O=$1 B=$2 S=$3; shift 3
export PATH=${RXM_RADIANCE}/bin:$PATH RADIANCE_HOME=${RXM_RADIANCE}/share/radiance RADIANCE_HOST_DEVICES=1
R=${O%.rad}.recipe; : > $R
for g in "$@"; do echo "$g exl3 src=$S codebook=mul1" >> $R; done
cat ${RXM_MODELS}/$B.recipe >> $R
cd ${RXM_CONVDIR:-/workspace}   # rad-convert reads models/qwen38-27b-bf16 (the bf16 checkpoint) from here
s=$(date +%s)
nice rad-convert models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
  --reuse ${RXM_MODELS}/q38-27b-$B.rad -o $O.part > ${O%.rad}.log 2>&1 && mv $O.part $O
rc=$?; echo "$(basename $O) rc=$rc $(( $(date +%s) - s ))s"; exit $rc
