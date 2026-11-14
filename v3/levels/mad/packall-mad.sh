#!/bin/bash
# packall-mad.sh HOME [N] [LEVELS...] -- (ds8) pack the mad ladder levels into /data/q27/packs-mad/ (#177-03, w151), as the
# lead's tools/packall.sh: rad-convert --no-error of the level's final.recipe (codebook=mad) with RADIANCE_HOME=HOME (a pack
# home whose exl3quant reads codebook=mad: qc-exl3-v3 >= e054e71, #177-02), host only, N at a time (default 2), ram.lock
# shared after MemAvailable >= 16 GiB; name as hostd: packs-mad/<level>[-s<scales8>].rad. Only levels DONE at the current
# scales (K <= 4.5 with scales) and with a local final/; LEVELS (K values) restricts and orders them (default: all, low K
# first). Stops before ds8 /data falls under 150 GB + the pack's size. Restartable (.part, existing packs skipped).
# Log mad/logs/packall.log. Process name `bash /data/q27/mad/tools/packall-mad.sh`.
set -u; Q=/data/q27; H=$1; N=${2:-2}; shift; [ $# -gt 0 ] && shift; R=$Q/ladder-mad; P=$Q/packs-mad; LG=$Q/mad/logs/packall.log
mkdir -p $P; [ -d $H ] || { echo "no home $H"; exit 1; }
export H P R Q LG
one() {
  local L=$R/q27v3-635bac02-k$1-mad n=q27v3-635bac02-k$1-mad s t
  [ -f $L/DONE ] && [ -d $L/final ] && [ -f $L/final.recipe ] || { echo "$(date -u +%T) skip K$1 (not ready)" >> $LG; return; }
  s=$($Q/venv/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); k=float(sys.argv[2]); s=d.get('scales'); sys.exit(1) if k <= 4.5 and not s else print((s or '')[:8])" $L/DONE $1) || { echo "$(date -u +%T) skip K$1 (swj not in)" >> $LG; return; }
  local out=$P/$n${s:+-s$s}.rad; [ -f $out ] && return
  local need=$(( $(du -sBG $L/final | cut -f1 | tr -dc 0-9) * 13 / 10 + 150 ))
  [ $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) -ge $need ] || { echo "$(date -u +%T) DISK: K$1 needs $need GB free: not packed" >> $LG; return; }
  ( flock -s 6; while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 16 ]; do sleep 20; done; t=$(date +%s)
    cd $Q && RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice -n 10 $Q/rad10/install/bin/rad-convert --no-error models/qwen38-27b-bf16 \
      --recipe $L/final.recipe --set quantization_config.quant_method=exl3 -o $out.part > $out.log 2>&1 \
      && mv $out.part $out && echo "$(date -u +%T) $out $(( $(date +%s) - t ))s" >> $LG || { rm -f $out.part; echo "$(date -u +%T) FAIL $out ($out.log)" >> $LG; } ) 6>$Q/locks/ram.lock
}
export -f one
KS=${*:-$(ls -d $R/q27v3-*-mad | sed 's/.*-k//; s/-mad$//' | sort -g)}
printf "%s\n" $KS | xargs -P $N -I{} bash -c "one {}"
echo "$(date -u +%T) PACKALL-MAD pass done ($(ls $P/*.rad 2>/dev/null | wc -l) packs)" >> $LG
