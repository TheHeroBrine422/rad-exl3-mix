#!/bin/bash
# donor144.sh NAME K CB -- (w144, #174-02, ds8) the trellis embedding donor /data/q27/emb/trellis/emb-NAME-K<k>.rad: the K0.125
# level pack re-converted with its own recipe plus `token_embd.weight exl3i src=<level>/src codebook=CB` (--reuse of the
# pack: only the embedding is imported; radmix takes only token_embd.weight from it), as w135's embll.sh built the Lloyd
# donors. Home /data/q27/w144/home (rad10-perf + this branch's exl3quant / qwen35_exl3 / exl3dense; its qwen35_exl3
# declares the bf16 vision tower at convert, so the donor carries model.visual.* too: unused). conv.lock + ram.lock
# shared. Restartable (.part, skips a finished donor).
set -eu
Q=/data/q27; D=$Q/emb/trellis; LV=q27v3-144dd6e1-k0.125-i8; B0=$(ls $Q/packs/$LV-s*.rad | head -n 1)
name=$1 k=$2 cb=$3
S=$D/$name/K$k/src; O=$D/emb-$name-K$k.rad
[ -f $O ] && exit 0
[ -f $S/q-embed.safetensors ] || { echo "no $S"; exit 1; }
R=$D/$name/K$k/donor.recipe; cp $Q/ladder/$LV/final.recipe $R
echo "token_embd.weight exl3i src=$S codebook=$cb" >> $R
rm -f $O.part; t=$(date +%s)
flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$Q/w144/home RADIANCE_HOST_DEVICES=1 nice -n 10 \
  $Q/rad10-perf/install/bin/rad-convert --no-error $Q/models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
  --reuse $B0 -o $O.part > $D/$name/K$k/donor.log 2>&1 || { echo "FAIL donor $name K$k ($D/$name/K$k/donor.log)"; exit 1; }
mv $O.part $O; echo "$O: $(( $(date +%s) - t )) s, $(stat -c %s $O) bytes"
