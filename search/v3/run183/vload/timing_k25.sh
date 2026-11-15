#!/bin/bash
# w162 (#185, step 4): the three ways to serve one whole candidate, timed on the same model (k2.5, all 336 trunk weights from
# the one level), ds8 card 7 one run at a time, golden-s173a:
#   mix     radmix writes the .rad (the search's path up to now), radiance reads the file
#   packs   the manifest with the level's pack as every trunk donor (no file written)
#   final   the manifest with the level's final/ as the trunk (no pack)
# Each run: the open (rad-info --bytes, the manifest's own open, without the GPU), the score's wall time and its load lines.
# The KLD rows of the three must be byte-identical. Usage: bash timing_k25.sh OUT   (the inputs are in the identity folder
# /data/q27/w162/id: k25.pack.mix and k25.final.mix, from final_mix.py; the final/ is /data/q27/w162/final-k25)
set -u
OUT=${1:-/data/q27/w162/timing}; ID=/data/q27/w162/id
BASE=/data/q27/srch183/base/base-T4.rad; GOLD=${GOLD:-/data/q27/w131/slices/golden-s173a}
RAD=/data/q27/w162/rad10/install/bin; W=/data/q27/w162; export RADIANCE_HOME=$W/rad10/home
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*"; }
grep -v "^base " "$ID/k25.pack.mix" > "$OUT/k25.mixfile"
flock -s /data/q27/locks/ram.lock nice -n 19 /data/q27/w158/tools/radmix "$OUT/k25.rad" $BASE "$OUT/k25.mixfile" > "$OUT/k25.radmix.log" 2>&1 \
  || { log "radmix FAILED"; exit 1; }
log "radmix: $(tail -1 "$OUT/k25.radmix.log")"
score() {   # score TAG PATH
  local tag=$1 path=$2 out=$OUT/$1
  rm -f "$out.json.part" "$out.json.part.rows" "$out.json" "$out.rows"
  flock /data/q27/locks/card7.lock flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=7 /usr/bin/time -f "%e" -o "$out.time" \
    "$RAD/radiance" --model "$path" --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 \
      --host-headroom-mib 8192 --kld-ref "$GOLD" --kld-out "$out.json.part" > "$out.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out.json.part.rows" ]; then log "$tag: score FAILED rc $rc ($out.log)"; return 1; fi
  mv "$out.json.part.rows" "$out.rows"; mv "$out.json.part" "$out.json"
  log "$tag: scored in $(tail -1 "$out.time") s"
}
for t in mix packs final; do
  case $t in
    mix) p=$OUT/k25.rad ;;
    packs) p=$ID/k25.pack.mix ;;
    final) p=$ID/k25.final.mix ;;
  esac
  /usr/bin/time -f "%e" -o "$OUT/$t.open.time" $RAD/rad-info --bytes "$p" > "$OUT/$t.open.bytes" 2>&1
  log "$t: open (rad-info --bytes) $(tail -1 "$OUT/$t.open.time") s"
  score $t "$p" || continue
done
for t in packs final; do
  if cmp -s "$OUT/mix.rows" "$OUT/$t.rows"; then log "$t: KLD rows SAME as mix ($(stat -c %s "$OUT/$t.rows") bytes)"
  else log "$t: KLD rows DIFFER from mix"; fi
done
rm -f "$OUT/k25.rad"
log done
