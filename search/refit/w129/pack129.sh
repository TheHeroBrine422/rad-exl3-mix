#!/bin/bash
# pack129.sh N LEVEL... -- pack the given final ladder levels that have no pack yet (as tools/packall.sh: rad-convert
# --no-error, packs/<level>[-s<scales8>].rad), N at a time; skips a level whose pack another rad-convert is writing.
# #169 (w129): packs only the 20 pass-1 levels the calibration swaps need (packing all 49 levels needs ~740 GB; ds8 had
# ~215 GB free on 11-12 10:50 PM CST). Log /data/q27/logs/pack129.log
cd /data/q27; N=$1; shift; export L=logs/pack129.log
one() { d=ladder/$1; [ -f $d/DONE ] || return 0
  s=$(/data/q27/venv/bin/python -c "import json; print(json.load(open(\"$d/DONE\")).get(\"scales\") or \"\")"); n=packs/$1${s:+-s${s:0:8}}.rad
  [ -f $n ] && return 0; pgrep -f "packs/$1[-.]" > /dev/null && return 0; t=$(date +%s)
  RADIANCE_HOME=/data/q27/rad/home RADIANCE_HOST_DEVICES=1 nice -n 10 /data/q27/rad/install/bin/rad-convert --no-error models/qwen38-27b-bf16 --recipe $d/final.recipe --set quantization_config.quant_method=exl3 -o $n.part > $n.log 2>&1 && mv $n.part $n && echo "$(date -u +%T) $n $(( $(date +%s) - t ))s" >> $L || { rm -f $n.part; echo "$(date -u +%T) FAIL $n" >> $L; }; }
export -f one
printf "%s\n" "$@" | xargs -P $N -I{} bash -c "one {}"
