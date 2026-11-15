#!/bin/bash
# #185 w173: the dry identity run on ds8: one live search mix, scored the mix way and the final/ way, card 7 under its lock.
#   bash dryrun.sh MIXNAME          (e.g. v3s183-6-pair-01568; its live rows must exist under srch183/scores)
#   A. live rows: the search's own rows for the mix (rad10-e157b, the live build; not re-run here)
#   B. mix path, candidate build: radmix writes the .rad (the live mix_cmd), radiance scores it on the same golden
#   C. final path, candidate build: the same mixfile with every trunk line of a READY level served from its final/
#      (subset_mix.py; READY = the level has FIN/LEVEL/READY, the rule hostd uses); the other trunk lines stay pack donors;
#      census_hybrid.py checks the manifest (as the candidate host would, before any scoring); radiance scores it
#   Rows must be byte-identical: A == B (the candidate build keeps the .rad path) and B == C (the final path).
#   The card lock is taken for each scoring run and released after it (flock -w: the search holds cards too).
set -u
M=${1:?usage: dryrun.sh MIXNAME}
W=${W:-/data/q27/w173}; OUT=$W/dry/$M; mkdir -p "$OUT"
MIX=/data/q27/srch183/mixes/$M.mix
GOLD=/data/q27/w169/slices/golden-s190a
BASE=/data/q27/srch183/base/base-T4.rad
FIN=/data/q27/finals-mad
RADMIX=/data/q27/w158/tools/radmix
RAD=$W/rad10/install/bin/radiance
export RADIANCE_HOME=$W/rad10/home
log() { echo "$(date -u +%FT%TZ) $M $*"; }
[ -f "$MIX" ] || { log "no mix $MIX"; exit 2; }
LIVE=/data/q27/srch183/scores/$M-s190a-m.json.rows
[ -f "$LIVE" ] || { log "no live rows $LIVE"; exit 2; }
cp "$LIVE" "$OUT/A.live.rows"

# the levels with a READY final/ (the rollout's result lines), as subset_mix.py names them: k7.5, k2.5, ...
LV=$(for f in "$FIN"/q27v3-635bac02-*-mad/READY; do [ -f "$f" ] || continue; d=$(basename "$(dirname "$f")"); d=${d#q27v3-635bac02-}; echo "${d%-mad}"; done | sort -V | tr '\n' ' ')
log "READY levels: $(echo $LV | wc -w)"
python3 -I "$W/tools/subset_mix.py" "$MIX" "$BASE" "$FIN" "$OUT/C.final.mix" $LV > "$OUT/C.subset.log" 2>&1 || { log "subset FAILED"; exit 2; }
cat "$OUT/C.subset.log"
# the census the candidate host runs on a final manifest (census_hybrid.py; the pack lines by census_manifest.py): before C is scored
tail -n +2 "$OUT/C.final.mix" > "$OUT/C.final.mixfile"
python3 -I "$W/tools/census_hybrid.py" "$OUT/C.final.mix" --mix "$OUT/C.final.mixfile" --base "$BASE" --final-root "$FIN" \
  --codebook mad --census "$W/code/store/census.py" --fast > "$OUT/C.census.log" 2>&1 \
  || { log "C: census_hybrid FAILED ($OUT/C.census.log)"; tail -3 "$OUT/C.census.log"; exit 2; }
log "C: $(tail -n 1 "$OUT/C.census.log")"
TRUNK=$(awk '$2 ~ /packs-mad\//' "$MIX" | wc -l)
FINL=$(awk '$2=="final"' "$OUT/C.final.mix" | wc -l)   # the trunk lines served from final/
log "trunk lines $TRUNK, served from final/ $FINL, from packs $((TRUNK - FINL))"

score() {   # score NAME MODEL  -> $OUT/NAME.rows (the card lock held for the run only)
  local n=$1 model=$2
  local t0=$(date +%s)
  flock -w 5400 /data/q27/locks/card7.lock flock -s -w 5400 /data/q27/locks/ram.lock \
    env ROCR_VISIBLE_DEVICES=7 "$RAD" --model "$model" --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 \
      --host-headroom-mib 8192 --kld-ref "$GOLD" --kld-out "$OUT/$n.json.part" > "$OUT/$n.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$n.json.part.rows" ]; then log "$n: score FAILED rc $rc ($OUT/$n.log)"; return 1; fi
  mv "$OUT/$n.json.part.rows" "$OUT/$n.rows"; mv "$OUT/$n.json.part" "$OUT/$n.json"
  log "$n: scored in $(( $(date +%s) - t0 )) s"
}

# radmix writes a ~8-12 GB .rad: ds8 must keep the floor after it (150 GiB, as the rollout does)
FLOOR=${FLOOR:-161061273600}; need=$((14 * 1000000000))
avail=$(df -B1 --output=avail "$W" | tail -n1)
if [ $((avail - need)) -lt "$FLOOR" ]; then log "STOP: $avail B free; the .rad needs ~14 GB above the floor"; exit 3; fi

log "B: radmix (mix path)"
flock -s -w 5400 /data/q27/locks/ram.lock nice -n 19 "$RADMIX" "$OUT/B.rad" "$BASE" "$MIX" > "$OUT/B.radmix.log" 2>&1 \
  || { log "radmix FAILED"; exit 2; }
score B "$OUT/B.rad" || exit 2
log "C: final path (manifest)"
score C "$OUT/C.final.mix" || exit 2
rm -f "$OUT/B.rad"                        # the mix file is not needed after its rows are scored

sha() { sha256sum "$1" | cut -c1-64; }
for x in A.live B C; do log "$x rows $(stat -c %s "$OUT/$x.rows") bytes sha256 $(sha "$OUT/$x.rows")"; done
r1=ok; r2=ok
cmp -s "$OUT/A.live.rows" "$OUT/B.rows" && log "A == B: SAME (the candidate build's .rad path is the live rows)" || { log "A != B: DIFFERENT"; r1=DIFF; }
cmp -s "$OUT/B.rows" "$OUT/C.rows" && log "B == C: SAME (final/ path rows are byte-identical to the mix path)" || { log "B != C: DIFFERENT"; r2=DIFF; }
echo "$M mix=$TRUNK trunk, final=$FINL, A==B $r1, B==C $r2" > "$OUT/summary.txt"
log "done"
