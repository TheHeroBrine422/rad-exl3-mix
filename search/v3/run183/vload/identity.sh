#!/bin/bash
# w162 (#185 step 3): identity of the virtual container (manifest) with the mix file radmix writes, on ds8.
#   For each mix (name -> mixfile):
#     1. reference: radmix (the search's own tool, w158) writes NAME.rad from BASE and the mixfile
#     2. manifest:  the same mixfile behind a `base BASE` line is NAME.mix (nothing is written for the candidate)
#     3. image:     rad-info --image-dump of the manifest, cmp against NAME.rad (directory, metadata, every weight)
#     4. KLD:       radiance --kld-ref GOLDEN on NAME.rad (kind mix), then on NAME.mix (kind virt), same build and card;
#                   cmp the rows files
#     5. the reference .rad is deleted after its last use (the mix file is what the change removes)
# Card: card 7 under /data/q27/locks/card7.lock, one run at a time (the lead's rule). Scratch: OUT.
#   bash identity.sh OUT [SCORED-NAME ...]     (default: the four scored mixes; pairdrop is cmp-only)
set -u
OUT=${1:-/data/q27/w162/ident}; shift || true
BASE=/data/q27/srch183/base/base-T4.rad
GOLD=${GOLD:-/data/q27/w131/slices/golden-s173a}
HOME_=${HOME_:-/data/q27/w162/rad10/home}
RAD=${RAD:-/data/q27/w162/rad10/install/bin}
M=/data/q27/srch183/mixes
mkdir -p "$OUT"
# the two synthetic mixes: the search mix with one ssm_ab replaced (scored), and with one weight dropped (cmp only)
# pairact: the same mix with blk.0.ssm_ab replaced by the donor the base was built from (same bytes) and its act line.
# (f16 on an ssm_ab is refused by the plugin at declare: "container format error", both paths alike.)
{ cat $M/v3s183-3.5-pair-00158.mix; echo "blk.0.ssm_ab.weight /data/q27/mad/ab/donor-ab-K4.rad"; echo "act blk.0.ssm_ab.weight a8g128"; } > "$OUT/pairact.mixfile"
{ cat $M/v3s183-3.5-pair-00158.mix; echo "drop blk.0.ssm_ab.weight"; } > "$OUT/pairdrop.mixfile"
declare -A MIXFILE=(
  [pair158]=$M/v3s183-3.5-pair-00158.mix
  [thompson86]=$M/v3s183-7-thompson-00086.mix
  [block13]=$M/v3s183-1.25-block-00013.mix
  [pairact]=$OUT/pairact.mixfile
  [pairdrop]=$OUT/pairdrop.mixfile
)
ORDER="pair158 thompson86 block13 pairact pairdrop"
SCORED=" pair158 thompson86 block13 pairact "      # pairdrop is cmp-only: a dropped ssm_ab may not load
[ "$#" -gt 0 ] && ORDER="$*"                      # a rerun of some names (scored only if they are in the scored set)
export RADIANCE_HOME=$HOME_
log() { echo "$(date -u +%FT%TZ) $*"; }

for n in $ORDER; do
  mf=${MIXFILE[$n]}; ref=$OUT/$n.rad; man=$OUT/$n.mix
  # 1. reference: radmix writes the mix file
  /usr/bin/time -f "radmix wall %e s" -o "$OUT/$n.radmix.time" nice -n 19 /data/q27/w158/tools/radmix "$ref" "$BASE" "$mf" \
      > "$OUT/$n.radmix.log" 2>&1 || { log "$n: radmix FAILED"; continue; }
  log "$n: radmix $(tail -1 "$OUT/$n.radmix.time")"
  # 2. the manifest: the same mixfile behind `base`
  { echo "base $BASE"; cat "$mf"; } > "$man"
  # 3. image identity: rad-info --image-dump writes the manifest's bytes as radmix lays them out
  if ! nice -n 19 "$RAD/rad-info" --image-dump "$OUT/$n.dump" "$man" > "$OUT/$n.dump.log" 2>&1; then
    log "$n: image dump FAILED ($(tail -1 "$OUT/$n.dump.log"))"; rm -f "$OUT/$n.dump" "$ref"; continue
  fi
  if cmp -s "$OUT/$n.dump" "$ref"; then log "$n: IMAGE SAME as radmix's $(stat -c %s "$ref") bytes"
  else log "$n: IMAGE DIFFERS: $(cmp "$OUT/$n.dump" "$ref" 2>&1 | head -1)"; fi
  rm -f "$OUT/$n.dump"
  # 4. KLD on both paths
  case "$SCORED" in *" $n "*)
    for kind in mix virt; do
      path=$([ $kind = mix ] && echo "$ref" || echo "$man")
      out=$OUT/$n.kld-$kind; rm -f "$out.json.part" "$out.json.part.rows" "$out.json" "$out.rows"
      flock /data/q27/locks/card7.lock env ROCR_VISIBLE_DEVICES=7 /usr/bin/time -f "%e" -o "$out.time" \
        "$RAD/radiance" --model "$path" --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 \
          --host-headroom-mib 8192 --kld-ref "$GOLD" --kld-out "$out.json.part" > "$out.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ] || [ ! -f "$out.json.part.rows" ]; then log "$n $kind: score FAILED rc $rc ($out.log)"; break; fi
      mv "$out.json.part.rows" "$out.rows"; mv "$out.json.part" "$out.json"
      log "$n $kind: scored in $(tail -1 "$out.time") s"
    done
    if [ -f "$OUT/$n.kld-mix.rows" ] && [ -f "$OUT/$n.kld-virt.rows" ]; then
      if cmp -s "$OUT/$n.kld-mix.rows" "$OUT/$n.kld-virt.rows"; then log "$n: KLD ROWS SAME ($(stat -c %s "$OUT/$n.kld-mix.rows") bytes)"
      else log "$n: KLD ROWS DIFFER"; fi
    fi
  ;; esac
  rm -f "$ref" "$ref.done"
done
log "done"
