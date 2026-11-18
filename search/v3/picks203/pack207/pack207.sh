#!/bin/bash
# pack207.sh SIZE... (box2) -- V3 release containers from the mad-only mixes (/data/q27/mad205/mixes/<size>-mad.mix):
#  1. trunk pack: rad-convert (w201 build, --no-error) with one recipe line per trunk weight,
#     "<weight> exl3i src=<finals-mad level>/final codebook=mad rate=<K>" (the hybrid mix's "final" lines, imported, not re-quantized)
#  2. container: radmix over the mix's base (base-T4.rad: norms, ssm_ab, ...) with the trunk weights from the pack, the head / embedding
#     donors and the act lines of the mix
#  3. census (full byte check of every weight vs its donor) -> /data/q27/release-v3/vram/<size>.rad (only on PASS)
# conv.lock + ram.lock shared (box2 rule). Restartable (.part; finished sizes skipped). Logs /data/q27/release-v3/work/<size>/.
set -u
Q=/data/q27; NEW=$Q/w201/rad21; MODEL=$Q/models/qwen38-27b-bf16; RADMIX=$Q/w158/tools/radmix; CEN=$Q/w158/code/census.py
R=$Q/release-v3; mkdir -p $R/vram $R/work
for S in "$@"; do
  M=$Q/mad205/mixes/$S-mad.mix; W=$R/work/$S; O=$R/vram/$S.rad; mkdir -p $W
  [ -f $O ] && { echo "$S: skip (have $O)"; continue; }
  [ "$(df --output=avail -B1G /data | tail -1)" -lt 150 ] && { echo "$S: STOP, /data < 150 GB free"; exit 1; }
  awk '$2=="final" {k=$3; sub(/.*-k/,"",k); sub(/-mad\/final$/,"",k); print $1" exl3i src="$3" codebook=mad rate="k}' $M > $W/trunk.recipe
  [ "$(wc -l < $W/trunk.recipe)" = 336 ] || { echo "$S: trunk recipe has $(wc -l < $W/trunk.recipe) lines"; continue; }
  B=$(awk '$1=="base" {print $2}' $M)
  if [ ! -f $W/trunk.rad ]; then
    rm -f $W/trunk.rad.part; t0=$(date +%s)
    ( cd $Q && flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$NEW/home RADIANCE_HOST_DEVICES=1 nice -n 10 \
      $NEW/install/bin/rad-convert --no-error $MODEL --recipe $W/trunk.recipe --set quantization_config.quant_method=exl3 -o $W/trunk.rad.part ) \
      > $W/convert.log 2>&1 || { echo "$S: FAIL convert ($W/convert.log)"; continue; }
    mv $W/trunk.rad.part $W/trunk.rad; echo "$S: trunk pack $(( $(date +%s) - t0 )) s"
  fi
  { awk -v P=$W/trunk.rad '$2=="final" {print $1" "P}' $M; grep -E '^(output|token_embd)\.weight ' $M; grep '^act ' $M; } > $W/release.mix
  rm -f $O.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock nice -n 10 $RADMIX $O.part $B $W/release.mix > $W/radmix.log 2>&1 \
    || { echo "$S: FAIL radmix ($W/radmix.log)"; continue; }
  if /data/q27/venv/bin/python -I $CEN $O.part --mix $W/release.mix --base $B --expect-act a8g128 --expect-emb trellis --max-version 4 \
       > $W/census.txt 2>&1; then
    mv $O.part $O; rm -f $W/trunk.rad; echo "$S: $O $(stat -c %s $O) bytes, census PASS"
  else echo "$S: CENSUS FAIL ($W/census.txt), kept $O.part"; fi
done
