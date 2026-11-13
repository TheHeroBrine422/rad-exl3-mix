#!/bin/bash
# embll.sh B... -- (w135, #174) Lloyd embedding donors at widths V2's containers do not carry (lead note 11-13: offer
# more of the embedding ladder): /data/q27/emb/emb-e<B>.rad = the K0.125 level pack re-converted with its own recipe
# plus `token_embd.weight rtn table=<B-bit Lloyd codebook> group=128 scale=bf16 transform=fwht128 rule=search` (V2's
# rule, served by embed_tq at any B in 1..8), --reuse of the pack so only the embedding is quantised. Codebooks:
# quantize/fitlevels.py on ds8's bf16 checkpoint (emb/ll/levels.txt; its 3..6-bit tables equal V2's levels.txt).
# conv.lock + ram.lock shared (as hd129.sh). Restartable (.part, skip finished).
set -eu
Q=/data/q27; D=$Q/emb; LV=q27v3-144dd6e1-k0.125-i8; B0=$(ls $Q/packs/$LV-s*.rad | head -n 1)
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $D/emb174.log; }
for B in "$@"; do
  O=$D/emb-e$B.rad; [ -f $O ] && continue
  T=$(awk -v b=$B '$1 == b {print $2}' $D/ll/levels.txt); [ -n "$T" ] || { log "no $B-bit codebook"; exit 1; }
  R=$D/ll/e$B.recipe; cp $Q/ladder/$LV/final.recipe $R
  echo "token_embd.weight rtn table=$T group=128 scale=bf16 transform=fwht128 rule=search" >> $R
  rm -f $O.part; t=$(date +%s)
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock env RADIANCE_HOME=$Q/rad10/home-pack RADIANCE_HOST_DEVICES=1 nice -n 10 \
    $Q/rad10/install/bin/rad-convert --no-error $Q/models/qwen38-27b-bf16 --recipe $R --set quantization_config.quant_method=exl3 \
    --reuse $B0 -o $O.part > $D/ll/e$B.log 2>&1 || { log "FAIL e$B ($D/ll/e$B.log)"; exit 1; }
  mv $O.part $O; log "$O: $B-bit Lloyd embedding, $(( $(date +%s) - t )) s"
done
