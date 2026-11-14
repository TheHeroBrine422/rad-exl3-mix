#!/bin/bash
# slim157.sh K... -- (w157, #174-03, box2 CPU) the eighth-step donors /data/q27/emb/trellis/emb-t-i8-K<k>.rad: w144's
# donor144.sh recipe (the K0.125 i8 level's final.recipe + `token_embd.weight exl3i src=<level>/src codebook=mul1i8`), its
# src paths moved to box2's /data/q27/ladder-src (the lead moved the ladder there 11-13; the importer checks src), --reuse
# of w144's K2.5 donor (the checkpoint-copied entries), then slimmed: radmix with every weight but token_embd.weight
# dropped (radmix's donors are read by name only), so a donor is ~0.2-0.8 GB instead of ~4.3 GB. The full container is
# deleted after the slim one is written. conv.lock + ram.lock shared. Restartable (.part, skips a finished donor).
# Byte check of the slim token_embd vs the full container's: embcmp157.py (log W/logs/slim157.jsonl).
set -u
Q=/data/q27 W=/data/q27/w157 D=/data/q27/emb/trellis I=/data/q27/rad10-perf/install
LV=/data/q27/ladder-src/q27v3-144dd6e1-k0.125-i8/final.recipe
mkdir -p $W/dtmp $W/logs
for k in "$@"; do
  S=$D/t-i8/K$k/src; O=$D/emb-t-i8-K$k.rad; F=$W/dtmp/full-K$k.rad
  [ -f $O ] && continue
  [ -f $S/q-embed.safetensors ] || { echo "K$k: no $S"; continue; }
  R=$D/t-i8/K$k/donor.recipe
  sed "s#/data/q27/ladder/#/data/q27/ladder-src/#" $LV > $R; echo "token_embd.weight exl3i src=$S codebook=mul1i8" >> $R
  if [ ! -f $F ]; then
    rm -f $F.part; t=$(date +%s)
    flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$W/b10e1/home RADIANCE_HOST_DEVICES=1 nice -n 10 \
      $I/bin/rad-convert --no-error $Q/models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
      --reuse $D/emb-t-i8-K2.5.rad -o $F.part > $D/t-i8/K$k/donor.log 2>&1 && mv $F.part $F \
      || { echo "K$k: FAIL convert ($D/t-i8/K$k/donor.log)"; continue; }
    echo "K$k: full $(( $(date +%s) - t )) s"
  fi
  python3 -I $W/tools/embcmp157.py --names $F | grep -vx token_embd.weight | sed 's/^/drop /' > $W/dtmp/slim-K$k.mix
  rm -f $O.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $W/tools/radmix $O.part $F $W/dtmp/slim-K$k.mix > $W/dtmp/slim-K$k.log 2>&1 \
    || { echo "K$k: FAIL radmix"; continue; }
  c=$(python3 -I $W/tools/embcmp157.py $O.part $F)
  echo "{\"K\": \"$k\", \"cmp\": $c}" >> $W/logs/slim157.jsonl
  echo "$c" | grep -q '"identical": true' && mv $O.part $O && rm -f $F && echo "K$k: donor $(stat -c %s $O) bytes" \
    || echo "K$k: FAIL token_embd differs after slimming"
done
