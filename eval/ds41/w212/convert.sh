#!/bin/bash
# w212 v2: the official DS4.1-F-1017 release: routed experts MXFP4 + Engram table fp8*e8m0[1x32] kept as released (native), rest of the trunk fp8 -> bf16 (exact)
export RADIANCE_HOME=/data/radiance-ds41/build/radiance_home
cd /data/q27/w212
exec /data/radiance-ds41/build/bin/rad-convert /data/ds41ck-full --arch deepseek41 --add engram-tables.safetensors \
  --tokenizer /data/ds41ck-full/tokenizer.json --recipe ds41-release.recipe --max-spec 0 --reuse ds41-official.rad \
  -o /data/q27/w212/ds41-official2.rad
