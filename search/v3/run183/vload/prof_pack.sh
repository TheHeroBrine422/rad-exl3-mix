#!/bin/bash
# w162 (#185 step 1): where rad-convert's time goes for one level, on ds8 (the pack host; the search's own pack_cmd
# with its environment). The level's per-tensor finals are copied to ds8 first (from box2's ladder-src-mad; see
# README). Phases, each timed:
#   floor   O_DIRECT read of the level's finals (floor_read.py: the disk's read ceiling, cache bypassed)
#   pack    rad-convert of the level, the same command as the host's eval.pack_cmd, with the finals' page cache dropped
#           first (cold reads); ram.lock shared (ds8's ram_mode), card 7 locked and the only device visible
#   wtest   O_DIRECT write of 6 GiB (the disk's write ceiling)
# Samples every second: the pack's CPU ticks (utime, stime), threads, RSS, rchar / wchar, read / write bytes; vmstat.
#   bash prof_pack.sh LEVEL SRCDIR OUTDIR     (SRCDIR = the copied final/ dir; OUTDIR = scratch)
set -u
L=${1:-q27v3-635bac02-k2.5-mad}; S=${2:-/data/q27/w162/finals/$L/final}; O=${3:-/data/q27/w162/prof}
mkdir -p "$O"; cd /data/q27
R=/data/q27/rad10-e157b
PY=/data/q27/venv/bin/python
sed "s|/data/q27/ladder-mad/$L/final|$S|g" /data/q27/ladder-mad/$L/final.recipe > "$O/$L.recipe"
echo "recipe: $(wc -l < "$O/$L.recipe") rules, src $(awk '{print $3}' "$O/$L.recipe" | sort -u)"

exec 7>/data/q27/locks/card7.lock; flock 7                   # GPU: one card, under its lock, short
exec 9>/data/q27/locks/ram.lock; flock -s 9                  # pack: ram.lock shared (ram_mode shared on ds8)
echo "locks held $(date -u +%FT%TZ)"

sec() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.2f", b - a}'; }
# --- floor: direct reads of the finals (cold: they were just copied, so drop their cache first) ---
$PY -I -c 'import os,sys,glob
for p in sys.argv[1:]:
    fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' $S/q-L*.safetensors
$PY -I "$(dirname "$0")/floor_read.py" $S/q-L*.safetensors | tee "$O/$L.floor"
t1=$(date +%s.%N)
dd if=/dev/zero of="$O/$L.wtest" bs=16M count=384 oflag=direct status=none; t2=$(date +%s.%N)
echo "floor write 6 GiB direct in $(sec $t1 $t2) s" | tee -a "$O/$L.floor"; rm -f "$O/$L.wtest"
$PY -I -c 'import os,sys,glob
for p in sys.argv[1:]:
    fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' $S/q-L*.safetensors

# --- pack ---
rm -f "$O/$L.rad" "$O/$L.rad.part"
vmstat 1 > "$O/$L.vmstat" 2>&1 & VM=$!
t3=$(date +%s.%N)
env RADIANCE_HOME=$R/home-pack RADIANCE_HOST_DEVICES=1 ROCR_VISIBLE_DEVICES=7 \
  /usr/bin/time -v -o "$O/$L.time" nice -n 19 $R/install/bin/rad-convert --no-error models/qwen38-27b-bf16 \
  --recipe "$O/$L.recipe" --set quantization_config.quant_method=exl3 -o "$O/$L.rad.part" > "$O/$L.log" 2>&1 &
PK=$!
C=""; for i in $(seq 1 50); do C=$(pgrep -n -f "install/bin/rad-convert --no-error" || true); [ -n "$C" ] && break; sleep 0.1; done
echo "t_sec utime stime threads rss_kib rchar wchar read_bytes write_bytes" > "$O/$L.samples"
while [ -n "$C" ] && [ -d /proc/$C ]; do
  s=$(awk '{print $14, $15}' /proc/$C/stat 2>/dev/null); io=$(awk '/^rchar|^wchar|^read_bytes|^write_bytes/{printf "%s ", $2}' /proc/$C/io 2>/dev/null)
  th=$(awk '/^Threads/{print $2}' /proc/$C/status 2>/dev/null); rss=$(awk '/^VmRSS/{print $2}' /proc/$C/status 2>/dev/null)
  echo "$(date +%s.%N) $s $th ${rss:-0} $io" >> "$O/$L.samples"; sleep 1
done
wait $PK; rc=$?; t4=$(date +%s.%N)
kill $VM 2>/dev/null
echo "pack rc $rc wall $(sec $t3 $t4) s"
if [ -f "$O/$L.rad.part" ]; then mv "$O/$L.rad.part" "$O/$L.rad"; ls -l "$O/$L.rad"; fi
grep -E "weights|finish|wrote|^E " "$O/$L.log" | sed 's/^/log: /'
awk '/User time|System time|Elapsed|Maximum resident|Major|Minor|Voluntary|Involuntary|File system/' "$O/$L.time"
