#!/bin/bash
# head197.sh CONTAINER TAG CLASS... -- #197 (w185): dequantise a KL-IQR row-split head on box2 (scratch cwd /data/q27/w185).
# Per output.kB class: its ids (raw i32, dq197 --raw) and its weight (bf16); plus output.vrow (raw i32). Restartable:
# existing files are skipped; every output is written as .part and renamed. The full head is then assembled by
# assemble197.py (run separately). Env: DQ (dq197 binary), PLUGIN (libquant.so of the container's build).
set -eu
C=$1; TAG=$2; shift 2
DQ=${DQ:-/data/q27/w185/build/dq197}; PLUGIN=${PLUGIN:-/data/q27/w177/install/share/radiance/quantizers/libquant.so}
cd /data/q27/w185; mkdir -p donors
for k in "$@"; do
  [ -s donors/$TAG-$k.ids ] || { nice -n 19 $DQ $PLUGIN $C output.$k.ids - --raw donors/$TAG-$k.ids.part && mv donors/$TAG-$k.ids.part donors/$TAG-$k.ids; }
  [ -s donors/$TAG-$k.bf16 ] || { nice -n 19 $DQ $PLUGIN $C output.$k.weight donors/$TAG-$k.bf16.part && mv donors/$TAG-$k.bf16.part donors/$TAG-$k.bf16; }
done
[ -s donors/$TAG-vrow.bin ] || { nice -n 19 $DQ $PLUGIN $C output.vrow - --raw donors/$TAG-vrow.bin.part && mv donors/$TAG-vrow.bin.part donors/$TAG-vrow.bin; }
echo "head197 $TAG classes done: $*"
