#!/bin/bash
# mkpack197.sh LEVEL -- #197 (w185): one level's pack for radmix, built with the search's pack_cmd on the level's own
# final/ (the recipe's src path is rewritten from ladder-mad to ladder-src-mad, the only one that exists on box2).
# Holds conv.lock and ram.lock exclusive, as the live search does for every rad-convert. Output: packs/LEVEL.rad
# (written as .part, then renamed; an existing output is skipped). Run on box2 from /data/q27/w185.
set -eu
L=$1; W=/data/q27/w185; OUT=$W/packs/$L.rad
mkdir -p $W/packs $W/recipes
[ -s $OUT ] && { echo "$L exists"; exit 0; }
R=$W/recipes/$L.recipe
sed "s#/data/q27/ladder-mad/#/data/q27/ladder-src-mad/#g" /data/q27/ladder-src-mad/$L/final.recipe > $R
rm -f $OUT.part
t0=$(date +%s)
flock /data/q27/locks/conv.lock flock /data/q27/locks/ram.lock bash -c "cd /data/q27 && RADIANCE_HOME=/data/q27/rad10-e157b/home-pack RADIANCE_HOST_DEVICES=1 /data/q27/rad10-e157b/install/bin/rad-convert --no-error models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 -o $OUT.part"
mv $OUT.part $OUT
echo "$L done in $(( $(date +%s) - t0 )) s, $(stat -c %s $OUT) bytes"
