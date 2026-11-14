#!/bin/bash
# hdonor173.sh PLAN... -- (w158, #173-03, ds8 CPU) served R16 head donors for the mad menu: per plan of hcurve151 (u<K> = mad
# R16s003mad-K<K>, u<K>m / u7 = mul1 R16s003m1-K<K>) rad-convert of w151's a/b donor recipe base (mad K1.375 level, local
# final/) + `output.weight exl3i src=<q-head dir> codebook=<mad|mul1> rate=K rows=r16`, --reuse of the K1.375 mad pack (trunk
# entries copied), then slimmed (radmix drop of every weight but output.weight; w157's slim157 form) ->
# /data/q27/w158/heads/hd-<plan>.rad. conv.lock + ram.lock shared. Restartable (.part; finished donors skipped).
set -u
Q=/data/q27 W=/data/q27/w158 E=/data/q27/rad10-e157b; H=$W/heads; mkdir -p $H/tmp $H/logs
BR=$Q/mad/ab/donor-ab-K4.recipe; PH=$Q/mad/rade05/home; RU=$Q/packs-mad/q27v3-635bac02-k1.375-mad-s6c57a3d1.rad
for p in "$@"; do
  O=$H/hd-$p.rad; F=$H/tmp/full-$p.rad; [ -f $O ] && continue
  case $p in
    *m) k=${p#u}; k=${k%m}; d=$Q/heads/q/R16s003m1/K$k; cb=mul1 ;;
    u7) k=7; d=$Q/heads/q/R16s003m1/K7; cb=mul1 ;;
    *)  k=${p#u}; d=$Q/heads/q/R16s003mad/K$k; cb=mad ;;
  esac
  [ -f $d/q-head.safetensors ] || { echo "$p: no $d/q-head.safetensors"; continue; }
  R=$H/tmp/$p.recipe; { grep -v '^output\.weight\|ssm_ab' $BR; echo "output.weight exl3i src=$d codebook=$cb rate=$k rows=r16"; } > $R
  if [ ! -f $F ]; then
    rm -f $F.part; t=$(date +%s)
    ( cd $Q && flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$PH RADIANCE_HOST_DEVICES=1 nice -n 10 \
      $Q/rad10-perf/install/bin/rad-convert --no-error models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
      --reuse $RU -o $F.part ) > $H/logs/$p.convert.log 2>&1 && mv $F.part $F || { echo "$p: FAIL convert ($H/logs/$p.convert.log)"; rm -f $F.part; continue; }
    echo "$p: full $(( $(date +%s) - t )) s"
  fi
  $Q/venv/bin/python -I $W/code/radnames.py $F | grep -vx output.weight | sed 's/^/drop /' > $H/tmp/slim-$p.mix
  rm -f $O.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $W/tools/radmix $O.part $F $H/tmp/slim-$p.mix > $H/logs/$p.slim.log 2>&1 \
    || { echo "$p: FAIL radmix ($H/logs/$p.slim.log)"; rm -f $O.part; continue; }
  if $Q/venv/bin/python -I $W/code/census.py $O.part --expect-head "exl3i $cb K$k r16" > $H/logs/$p.census.txt 2>&1; then
    mv $O.part $O && rm -f $F && echo "$p: donor $(stat -c %s $O) bytes"
  else echo "$p: FAIL census ($H/logs/$p.census.txt)"; rm -f $O.part; fi
done
