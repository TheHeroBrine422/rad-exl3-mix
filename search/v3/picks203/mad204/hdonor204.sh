#!/bin/bash
# hdonor204.sh K... -- (lead 11-17, mad-only check) served R16 MAD head donors at K >= 7 (where u<K> names are mul1),
# from ds8 heads/q/R16s003mad/K<K> -> /data/q27/w158/heads/hd-u<K>a.rad ("a" = mad). hdonor173.sh without --reuse (the
# K1.375 mad pack is deleted): base recipe donor-ab-K4 with its trunk source pointed at finals-mad K1.375, then slimmed
# to output.weight only. Restartable (.part).
set -u
Q=/data/q27 W=/data/q27/w158; H=$W/heads; mkdir -p $H/tmp $H/logs
BR=$Q/mad/ab/donor-ab-K4.recipe; PH=$Q/mad/rade05/home
OLD=$Q/mad/ab/base-q27v3-635bac02-k1.375-mad/final; NEW=$Q/finals-mad/q27v3-635bac02-k1.375-mad/final
for k in "$@"; do
  p=u${k}a O=$H/hd-$p.rad F=$H/tmp/full-$p.rad d=$Q/heads/q/R16s003mad/K$k; [ -f $O ] && continue
  [ -f $d/q-head.safetensors ] || { echo "$p: no $d/q-head.safetensors"; continue; }
  R=$H/tmp/$p.recipe; { grep -v '^output\.weight\|ssm_ab' $BR | sed "s#$OLD#$NEW#"; echo "output.weight exl3i src=$d codebook=mad rate=$k rows=r16"; } > $R
  if [ ! -f $F ]; then
    rm -f $F.part; t=$(date +%s)
    ( cd $Q && flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$PH RADIANCE_HOST_DEVICES=1 nice -n 10 \
      $Q/rad10-perf/install/bin/rad-convert --no-error models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
      -o $F.part ) > $H/logs/$p.convert.log 2>&1 && mv $F.part $F || { echo "$p: FAIL convert ($H/logs/$p.convert.log)"; rm -f $F.part; continue; }
    echo "$p: full $(( $(date +%s) - t )) s"
  fi
  $Q/venv/bin/python -I $W/code/radnames.py $F | grep -vx output.weight | sed 's/^/drop /' > $H/tmp/slim-$p.mix
  rm -f $O.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $W/tools/radmix $O.part $F $H/tmp/slim-$p.mix > $H/logs/$p.slim.log 2>&1 \
    || { echo "$p: FAIL radmix ($H/logs/$p.slim.log)"; rm -f $O.part; continue; }
  if $Q/venv/bin/python -I $W/code/census.py $O.part --expect-head "exl3i mad K$k r16" > $H/logs/$p.census.txt 2>&1; then
    mv $O.part $O && rm -f $F && echo "$p: donor $(stat -c %s $O) bytes"
  else echo "$p: FAIL census ($H/logs/$p.census.txt)"; rm -f $O.part; fi
done
