#!/bin/bash
# donor151.sh K BASE_K -- (ds8) #177-03: the mad a | b donor /data/q27/mad/ab/donor-ab-K<K>.rad as w147's donor147.sh: the
# exl3quant source of ab151.sh's rxq outputs (absrc147.py: mad/ab/K<K> -> mad/ab/absrc-K<K>, codebook mad), then the mad level
# BASE_K's pack re-converted with its own recipe (src -> a hard-link copy of its final/, so move-mad.sh may drop the original)
# plus `blk.*.ssm_ab.weight exl3i src=absrc-K<K> codebook=mad rate=<K>` (--reuse of the pack; radmix takes only the 48 ssm_ab
# weights from the donor). Home: the e054e71 pack home (mad importer; qwen35_exl3 maps ssm_ab). conv.lock + ram.lock shared.
# Restartable (.part; a finished donor is skipped). (w151)
set -eu
Q=/data/q27; A=$Q/mad/ab; H=$Q/mad/rade05/home; k=$1; BK=$2; LV=q27v3-635bac02-k$BK-mad; L=$Q/ladder-mad/$LV
B0=$(ls $Q/packs-mad/$LV*.rad | head -n 1); S=$A/absrc-K$k; O=$A/donor-ab-K$k.rad; BF=$A/base-$LV
[ -f $O ] && exit 0
[ -d $BF/final ] || { mkdir -p $BF; cp -al $L/final $BF/final; cp $L/final.recipe $BF/final.recipe; }
[ -f $S/model.safetensors ] || $Q/venv/bin/python $Q/mad/tools/absrc147.py $A/K$k $BF/final $S
R=$A/donor-ab-K$k.recipe
sed "s|src=$L/final|src=$BF/final|" $BF/final.recipe > $R
echo "blk.*.ssm_ab.weight exl3i src=$S codebook=mad rate=$k" >> $R
rm -f $O.part; t=$(date +%s)
flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice -n 10 \
  $Q/rad10-perf/install/bin/rad-convert --no-error $Q/models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
  --reuse $B0 -o $O.part > $A/donor-ab-K$k.log 2>&1 || { echo "FAIL donor K$k ($A/donor-ab-K$k.log)"; exit 1; }
mv $O.part $O; echo "$O: $(( $(date +%s) - t )) s, $(stat -c %s $O) bytes"
