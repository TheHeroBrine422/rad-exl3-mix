#!/bin/bash
# w212: convert the official DS4.1-F-1017 release (MXFP4 experts kept, trunk fp8 -> bf16 cast, Engram table bf16) for bakon radiance
export RADIANCE_HOME=/data/radiance-ds41/build/radiance_home
cd /data/q27/w212
exec /data/radiance-ds41/build/bin/rad-convert /data/ds41ck-full --arch deepseek41 --add engram-tables.safetensors \
  --tokenizer /data/ds41ck-full/tokenizer.json --recipe ds41-release-bf16trunk.recipe --max-spec 0 \
  -o /data/q27/w212/ds41-official.rad
