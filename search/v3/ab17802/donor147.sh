#!/bin/bash
# donor147.sh K -- (w147, #178-02, ds8) the a | b donor W/donor-ab-K<K>.rad: the K0.125 level pack re-converted with its own
# recipe plus `blk.*.ssm_ab.weight exl3i src=W/absrc-K<K> codebook=mul1i8 rate=<K>` (--reuse of the pack, as w144's
# donor144.sh; radmix takes only the 48 ssm_ab weights from it). Home W/home (rad10-perf + this branch's exl3quant /
# qwen35_exl3: the plugin maps ssm_ab and, the recipe naming it exl3i, declares it instead of plain ssm_a / ssm_b).
# conv.lock + ram.lock shared. Restartable (.part, skips a finished donor).
set -eu
Q=/data/q27; W=$Q/w147; LV=q27v3-144dd6e1-k0.125-i8; B0=$(ls $Q/packs/$LV-s*.rad | head -n 1)
k=$1; S=$W/absrc-K$k; O=$W/donor-ab-K$k.rad
[ -f $O ] && exit 0
[ -f $S/model.safetensors ] || { echo "no $S"; exit 1; }
R=$W/donor-ab-K$k.recipe; cp $Q/ladder/$LV/final.recipe $R
echo "blk.*.ssm_ab.weight exl3i src=$S codebook=mul1i8 rate=$k" >> $R
rm -f $O.part; t=$(date +%s)
flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$W/home RADIANCE_HOST_DEVICES=1 nice -n 10 \
  $Q/rad10-perf/install/bin/rad-convert --no-error $Q/models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
  --reuse $B0 -o $O.part > $W/donor-ab-K$k.log 2>&1 || { echo "FAIL donor K$k ($W/donor-ab-K$k.log)"; exit 1; }
mv $O.part $O; echo "$O: $(( $(date +%s) - t )) s, $(stat -c %s $O) bytes"
