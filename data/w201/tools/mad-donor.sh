#!/bin/bash
# mad-donor.sh K... -- (w201, box2) the mad token-embedding donors /data/q27/emb/trellis/emb-t-mad-K<k>.rad, made with the
# w201 build's rad-convert the way the i8 donors were (w144 donor144.sh, w157 slim157.sh):
#   recipe = the nine trunk lines of the mad level K (finals-mad, imported as trellis, not quantised, codebook=mad rate=K)
#            + `token_embd.weight exl3i src=/data/q27/emb/trellis/mad/K<k>/src codebook=mad rate=<k>`
#   full container: rad-convert --no-error (no --reuse: nothing else is kept)
#   slim: radmix with every weight but token_embd.weight dropped (radmix reads donors by name); the slim token_embd must be
#         byte-identical to the full one's (embcmp157.py), else the donor is not written.
# conv.lock exclusive + ram.lock shared around the convert (box2 rules). Restartable: skips a finished donor, .part files
# are redone. Logs: /data/q27/w201/donors/K<k>/{convert,slim}.log, summary /data/q27/w201/donors/summary.jsonl.
set -u
NEW=/data/q27/w201/rad21
Q=/data/q27; D=$Q/emb/trellis; MODEL=$Q/models/qwen38-27b-bf16
RADMIX=$Q/w158/tools/radmix; CMP=$Q/w157/tools/embcmp157.py
SUM=/data/q27/w201/donors/summary.jsonl
mkdir -p /data/q27/w201/donors
for K in "$@"; do
  W=/data/q27/w201/donors/K$K; O=$D/emb-t-mad-K$K.rad; S=$D/mad/K$K/src; L=$Q/finals-mad/q27v3-635bac02-k$K-mad/final
  mkdir -p $W
  [ -f $O ] && { echo "K$K: skip (have $O)"; continue; }
  [ -f $S/q-embed.safetensors ] || { echo "K$K: no $S/q-embed.safetensors"; continue; }
  [ -d $L ] || { echo "K$K: no level $L"; continue; }
  R=$W/donor.recipe
  # the trunk lines of the mad-packs K4 recipe, moved to this level and rate
  sed -e "s#/data/q27/mad-packs/q27v3-635bac02-k4-mad/final#$L#" -e "s# rate=4\$# rate=$K#" $Q/mad-packs/q27v3-635bac02-k4-mad.recipe > $R
  echo "token_embd.weight exl3i src=$S codebook=mad rate=$K" >> $R
  n=$(grep -c "^blk\..*rate=$K\$" $R); [ "$n" -eq 9 ] || { echo "K$K: recipe has $n trunk lines at rate $K (want 9)"; continue; }
  t0=$(date +%s)
  if [ ! -f $W/full.rad ]; then
    rm -f $W/full.rad.part
    flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$NEW/home RADIANCE_HOST_DEVICES=1 nice -n 10 \
      $NEW/install/bin/rad-convert --no-error $MODEL --recipe $R --set quantization_config.quant_method=exl3 \
      -o $W/full.rad.part > $W/convert.log 2>&1 || { echo "K$K: FAIL convert ($W/convert.log)"; continue; }
    mv $W/full.rad.part $W/full.rad
  fi
  t1=$(date +%s)
  python3 -I $CMP --names $W/full.rad | grep -vx token_embd.weight | sed 's/^/drop /' > $W/slim.mix
  rm -f $O.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock nice -n 10 $RADMIX $O.part $W/full.rad $W/slim.mix > $W/slim.log 2>&1 \
    || { echo "K$K: FAIL radmix ($W/slim.log)"; continue; }
  c=$(python3 -I $CMP $O.part $W/full.rad)
  if echo "$c" | grep -q '"identical": true'; then
    mv $O.part $O && rm -f $W/full.rad
    echo "{\"K\": \"$K\", \"donor\": \"$O\", \"bytes\": $(stat -c %s $O), \"convert_s\": $((t1 - t0)), \"cmp\": $c}" >> $SUM
    echo "K$K: donor $(stat -c %s $O) bytes, convert $((t1 - t0)) s"
  else
    echo "K$K: FAIL token_embd differs after slimming ($W)"; echo "$c" > $W/cmp.json.bad
  fi
done
