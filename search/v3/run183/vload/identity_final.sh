#!/bin/bash
# w162 (#185, step 4): KLD identity of the final/ path (B) against the mix path, on ds8.
#   For each scored mix (pair158, thompson86, block13, pairact; the same mixes as identity.sh):
#     1. subset_mix.py: the mix with the trunk lines of three levels (k1, k3.75, k7) served from their final/ directories
#        (`<weight> final DIR codebook=mad`), every other line as it is (head, embedding, a/b donor, act lines)
#     2. radiance --model NAME.final.mix on golden-s173a (card 7, one run at a time), rows compared byte for byte with the
#        mix path's rows from identity.sh (the same mix as radmix writes it: ds8 $IDENT/NAME.kld-mix.rows)
#   Control: the mix path's pair158 through this build (radmix writes pair158.rad; rows vs the same reference).
#   bash identity_final.sh OUT          (scratch OUT; the finals are the three levels under $FIN)
set -u
OUT=${1:-/data/q27/w162/idb}
BASE=/data/q27/srch183/base/base-T4.rad
GOLD=${GOLD:-/data/q27/w131/slices/golden-s173a}
RAD=${RAD:-/data/q27/w162/rad10/install/bin}
HOME_=${HOME_:-/data/q27/w162/rad10/home}
FIN=${FIN:-/data/q27/w162/finals-sub}
IDENT=${IDENT:-/data/q27/w162/ident}
M=/data/q27/srch183/mixes
W=/data/q27/w162
mkdir -p "$OUT"
export RADIANCE_HOME=$HOME_
log() { echo "$(date -u +%FT%TZ) $*"; }
declare -A SRC=(
  [pair158]=$M/v3s183-3.5-pair-00158.mix
  [thompson86]=$M/v3s183-7-thompson-00086.mix
  [block13]=$M/v3s183-1.25-block-00013.mix
  [pairact]=$IDENT/pairact.mixfile
)
score() {   # score NAME MANIFEST_OR_RAD  -> $OUT/NAME.$2.rows
  local n=$1 path=$2 tag=$3
  local out=$OUT/$n.$tag
  rm -f "$out.json.part" "$out.json.part.rows" "$out.json" "$out.rows"
  flock /data/q27/locks/card7.lock flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=7 /usr/bin/time -f "%e" -o "$out.time" \
    "$RAD/radiance" --model "$path" --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 \
      --host-headroom-mib 8192 --kld-ref "$GOLD" --kld-out "$out.json.part" > "$out.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out.json.part.rows" ]; then log "$n $tag: score FAILED rc $rc ($out.log)"; return 1; fi
  mv "$out.json.part.rows" "$out.rows"; mv "$out.json.part" "$out.json"
  log "$n $tag: scored in $(tail -1 "$out.time") s"
}
for n in pair158 thompson86 block13 pairact; do
  python3 $W/subset_mix.py "${SRC[$n]}" $BASE $FIN "$OUT/$n.final.mix" k1 k3.75 k7 || { log "$n: subset FAILED"; continue; }
  score $n "$OUT/$n.final.mix" final || continue
  if cmp -s "$OUT/$n.final.rows" "$IDENT/$n.kld-mix.rows"; then
    log "$n: KLD ROWS SAME as the mix path ($(stat -c %s "$OUT/$n.final.rows") bytes)"
  else
    log "$n: KLD ROWS DIFFER from the mix path"
  fi
done
# control: the mix path through this build (pair158, radmix writes the .rad, then the same score)
flock -s /data/q27/locks/ram.lock nice -n 19 /data/q27/w158/tools/radmix "$OUT/pair158-ctl.rad" $BASE "${SRC[pair158]}" > "$OUT/pair158-ctl.radmix.log" 2>&1 \
  || log "pair158 control: radmix FAILED"
if [ -f "$OUT/pair158-ctl.rad" ]; then
  score pair158 "$OUT/pair158-ctl.rad" ctl
  cmp -s "$OUT/pair158.ctl.rows" "$IDENT/pair158.kld-mix.rows" && log "pair158 control: KLD ROWS SAME as the mix path" \
    || log "pair158 control: KLD ROWS DIFFER"
  rm -f "$OUT/pair158-ctl.rad"
fi
log "done"
