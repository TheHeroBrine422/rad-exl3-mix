#!/bin/bash
# w162 (#185, step 4): the plane identity of one level, ds8, no GPU. For LEVEL (e.g. q27v3-635bac02-k7-mad):
#   1. copy its final/ from box2 (ladder-src-mad, the lead's per-tensor output) to $W/finals-plane/LEVEL/final
#   2. final_mix.py: the 336 trunk names and the manifest that serves all of them from that final/ (+ head / embedding donors)
#   3. rad-info --weights-dump on the level's pack (reference) and on the manifest (the names only), then cmp_dumps.py:
#      every trunk weight's header and plane bytes must be identical
#   4. delete the copy and the dumps (kept: the log and the names / manifest)
# Usage: BOX2_HOST=... BOX2_KEY=... bash plane_level.sh OUT LEVEL...   (box2's address and the ds8 key for it: local/ notes, not here)
set -u
OUT=$1; shift
W=/data/q27/w162; BASE=/data/q27/srch183/base/base-T4.rad
HEAD=/data/q27/w158/heads/hd-u3.625.rad; EMB=/data/q27/emb/trellis/emb-t-i8-K2.625.rad
RI=$W/rad10/install/bin/rad-info; export RADIANCE_HOME=$W/rad10/home
: "${BOX2_HOST:?set BOX2_HOST (local/ notes)}"; : "${BOX2_KEY:?set BOX2_KEY (local/ notes)}"
K="ssh -i $BOX2_KEY -o StrictHostKeyChecking=no -o Compression=no -c aes128-gcm@openssh.com"
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*"; }
for L in "$@"; do
  P=/data/q27/packs-mad/$L-s6c57a3d1.rad; [ -f "$P" ] || P=/data/q27/packs-mad/$L.rad
  [ -f "$P" ] || { log "$L: no pack"; continue; }
  D=$W/finals-plane/$L/final; mkdir -p "$D"
  nice -n 10 rsync -a --whole-file -e "$K" root@$BOX2_HOST:/data/q27/ladder-src-mad/$L/final/ "$D/" || { log "$L: copy FAILED"; continue; }
  nice -n 10 python3 $W/final_mix.py "$D" $BASE $HEAD $EMB "$OUT/$L" --pack "$P" > /dev/null || { log "$L: final_mix FAILED"; continue; }
  flock -s /data/q27/locks/ram.lock nice -n 19 $RI --weights-dump "$OUT/$L.ref.dump" --only "$OUT/$L.names" "$P" > "$OUT/$L.ref.log" 2>&1 || { log "$L: ref dump FAILED"; continue; }
  flock -s /data/q27/locks/ram.lock nice -n 19 $RI --weights-dump "$OUT/$L.b.dump" --only "$OUT/$L.names" "$OUT/$L.final.mix" > "$OUT/$L.b.log" 2>&1 || { log "$L: final dump FAILED"; continue; }
  nice -n 19 python3 $W/cmp_dumps.py "$OUT/$L.ref.dump" "$OUT/$L.b.dump" > "$OUT/$L.cmp.log" 2>&1; rc=$?
  log "$L: $(tail -1 "$OUT/$L.cmp.log") ($(grep -o 'identical weights: [0-9]* of [0-9]*' "$OUT/$L.cmp.log"))"
  rm -f "$OUT/$L.ref.dump" "$OUT/$L.b.dump"; rm -rf "$W/finals-plane/$L"
done
log done
