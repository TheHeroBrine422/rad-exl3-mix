#!/bin/bash
# #185 w173 (ds8): roll one mad level's final/ over from box2 and check its planes against its pack. No GPU.
#   BOX2_HOST=... BOX2_KEY=... bash rollout_level.sh LEVEL
#   (box2's address and the ds8 key for it come from the environment only; they are not written to any file here)
#   1. copy box2 ladder-src-mad/LEVEL/final/ -> $FIN/LEVEL/final.part (rsync, resumable); the md5 of every file must match
#      on both ends; then rename to $FIN/LEVEL/final.
#   2. planes: final_mix.py writes the manifest that serves the 336 trunk weights from that final/; rad-info --weights-dump
#      of the pack and of the manifest stream into two FIFOs (nothing written to disk); cmp_dumps.py: every weight's header
#      and every plane's bytes must be equal ("identical weights: 336 of 336", then IDENTICAL).
#   Exit: 0 READY (final/ kept), 1 REJECTED (planes differ: final/ moved to $W/rejected), 2 error (nothing recorded; a
#   rerun resumes), 3 stop (the copy would take the free space below FLOOR; nothing copied).
#   The result line $W/results/LEVEL.tsv is written last; a level with one is never run again.
set -u
L=${1:?usage: rollout_level.sh LEVEL}
W=${W:-/data/q27/w173}; FIN=${FIN:-/data/q27/finals-mad}; PK=${PK:-/data/q27/packs-mad}
T=$W/tools; RES=$W/results; PL=$W/planes
BASE=/data/q27/srch183/base/base-T4.rad
HEAD=/data/q27/w158/heads/hd-u3.625.rad
EMB=/data/q27/emb/trellis/emb-t-i8-K2.625.rad
RI=/data/q27/w162/rad10/install/bin/rad-info           # w162's build of the vload branch (e29c765b + 64735054)
export RADIANCE_HOME=/data/q27/w162/rad10/home
FLOOR=${FLOOR:-161061273600}                            # 150 GiB in bytes: the free space ds8 keeps
SRC=/data/q27/ladder-src-mad/$L/final
P=$PK/$L-s6c57a3d1.rad; [ -f "$P" ] || P=$PK/$L.rad     # the pack the live mixes name (13 levels, K 4.625 and up, have no stamp)
K=$(echo "$L" | sed -E 's/^.*-k([0-9.]+)-mad$/\1/')
: "${BOX2_HOST:?set BOX2_HOST}"; : "${BOX2_KEY:?set BOX2_KEY}"
SSHO="-i $BOX2_KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
rssh() { ssh $SSHO root@"$BOX2_HOST" "$@"; }
say() { echo "$(date -u +%FT%TZ) $L $*"; }
mkdir -p "$FIN/$L" "$RES" "$PL" "$W/rejected/$L" "$W/tmp"

# READY marker (the hybrid hostd serves a level with it from final/, every other level from its pack) and the verified md5
# list the census checks (FIN/LEVEL/final.md5). Written after the result line, so a marker never exists without a READY result.
finalize() {
  [ -s "$RES/$L.md5.ds8" ] || { say "no md5 list to finalize"; exit 2; }
  cp "$RES/$L.md5.ds8" "$FIN/$L/final.md5.part" && mv "$FIN/$L/final.md5.part" "$FIN/$L/final.md5"
  { echo READY; cat "$RES/$L.tsv"; } > "$FIN/$L/READY.part" && mv "$FIN/$L/READY.part" "$FIN/$L/READY"
  say "READY marker written"
}
if [ -f "$RES/$L.tsv" ]; then
  st=$(cut -f8 "$RES/$L.tsv"); say "done before: $st"
  if [ "$st" = READY ]; then [ -f "$FIN/$L/READY" ] || finalize; exit 0; fi
  exit 1
fi
[ -f "$P" ] || { say "no pack $P"; exit 2; }
[ -f "$T/final_mix.py" ] && [ -f "$T/cmp_dumps.py" ] || { say "tools missing under $T"; exit 2; }
sums() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 nice -n 19 md5sum); }

if [ ! -d "$FIN/$L/final" ]; then
  have=0; [ -d "$FIN/$L/final.part" ] && have=$(du -sb "$FIN/$L/final.part" | cut -f1)
  need=$(rssh "du -sb $SRC" | cut -f1); [ -n "$need" ] || { say "box2 unreachable"; exit 2; }
  avail=$(df -B1 --output=avail "$FIN" | tail -n1)
  if [ $((avail - (need - have))) -lt "$FLOOR" ]; then
    say "STOP: copy of $((need - have)) B, free $avail B, floor $FLOOR B"; exit 3
  fi
  say "copy $need B (have $have B) -> $FIN/$L/final.part"
  mkdir -p "$FIN/$L/final.part"
  nice -n 19 ionice -c2 -n7 rsync -a --partial --timeout=900 -e "ssh $SSHO" root@"$BOX2_HOST":"$SRC"/ "$FIN/$L/final.part"/ \
    || { say "COPY FAILED rc $?"; exit 2; }
  say "md5 both ends"
  rssh "cd $SRC && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 md5sum" > "$RES/$L.md5.box2" &
  p0=$!
  sums "$FIN/$L/final.part" > "$RES/$L.md5.ds8"
  wait $p0; rc0=$?
  if [ $rc0 != 0 ] || [ ! -s "$RES/$L.md5.box2" ] || ! cmp -s "$RES/$L.md5.box2" "$RES/$L.md5.ds8"; then
    say "COPY MISMATCH (box2 rc $rc0); final.part removed"; rm -rf "$FIN/$L/final.part"; exit 2
  fi
  mv "$FIN/$L/final.part" "$FIN/$L/final"
  say "copy verified: $(wc -l < "$RES/$L.md5.ds8") files"
fi
D=$FIN/$L/final
bytes=$(find "$D" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
files=$(wc -l < "$RES/$L.md5.ds8")
digest=$(md5sum < "$RES/$L.md5.ds8" | cut -c1-32)

say "planes: manifest"
t0=$(date +%s)
python3 -I "$T/final_mix.py" "$D" "$BASE" "$HEAD" "$EMB" "$PL/$L" --pack "$P" > "$PL/$L.mix.log" 2>&1 \
  || { say "final_mix FAILED ($PL/$L.mix.log)"; exit 2; }
F=$W/tmp/$L; rm -f "$F.ref" "$F.fin"; mkfifo "$F.ref" "$F.fin"
flock -s -w 3600 /data/q27/locks/ram.lock nice -n 19 ionice -c2 -n7 "$RI" --weights-dump "$F.ref" --only "$PL/$L.names" "$P" \
  > "$PL/$L.ref.log" 2>&1 &
p1=$!
flock -s -w 3600 /data/q27/locks/ram.lock nice -n 19 ionice -c2 -n7 "$RI" --weights-dump "$F.fin" --only "$PL/$L.names" "$PL/$L.final.mix" \
  > "$PL/$L.fin.log" 2>&1 &
p2=$!
timeout 5400 nice -n 19 python3 -I "$T/cmp_dumps.py" "$F.ref" "$F.fin" > "$PL/$L.cmp.log" 2>&1; crc=$?
wait $p1; r1=$?; wait $p2; r2=$?
rm -f "$F.ref" "$F.fin"
secs=$(( $(date +%s) - t0 ))

ok=0
if [ $crc = 0 ] && [ $r1 = 0 ] && [ $r2 = 0 ] && grep -q "identical weights: 336 of 336" "$PL/$L.cmp.log" \
   && [ "$(tail -n1 "$PL/$L.cmp.log")" = IDENTICAL ]; then ok=1; fi
if [ $ok = 1 ]; then planes="IDENTICAL 336/336"; status=READY
else planes="DIFFERENT (cmp $crc, rad-info $r1/$r2)"; status=REJECTED; fi
say "planes: $planes (${secs}s)"

if [ $ok != 1 ]; then mv "$D" "$W/rejected/$L/final"; say "REJECTED: final/ moved to $W/rejected/$L/final"; fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$L" "$K" "$bytes" "$files" "VERIFIED" "$digest" "$planes" "$status" \
  "$(date -u +%FT%TZ)" "$secs" "$(basename "$P")" > "$RES/$L.tsv.part"
mv "$RES/$L.tsv.part" "$RES/$L.tsv"
if [ $ok = 1 ]; then finalize; exit 0; fi
exit 1
