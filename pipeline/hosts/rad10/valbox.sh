#!/bin/bash
# valbox.sh CARD -- #170 on a second host (box2) with the rad10 tree copied from ds8 (same paths): exl3i's device tests
# (FUSE off) and the KLD of V2 3.00 and the K4 pack (W/val/packs/K4-r7.rad = ds8's 7.14 pack) on golden-ver3-8k with the
# 10.1 build, rows to compare with ds8's (kldcmp-hosts). Card lock, ram.lock shared. Restartable. Log W/val/valbox.log. (w130)
set -u; CARD=$1; W=/data/q27/rad10; Q=/data/q27; V=$W/val; R=$V/res; L=$Q/locks; G=$Q/goldens/golden-ver3-8k
export PATH=$Q/venv/bin:$PATH; mkdir -p $R; exec >> $V/valbox.log 2>&1; ulimit -n 65536
say() { echo "$(date -u +%T) $*"; }
exec 8>$L/card$CARD.lock; flock -x 8; say "card$CARD.lock held"
if [ ! -f $R/t-exl3i-gpu.txt ]; then ROCR_VISIBLE_DEVICES=$CARD ctest --test-dir $W/pb/exl3i -L gpu -E test_twopass_fuse --output-on-failure --timeout 900 > $R/t-exl3i-gpu.txt.part 2>&1; mv $R/t-exl3i-gpu.txt.part $R/t-exl3i-gpu.txt; say "exl3i gpu: $(grep 'tests passed' $R/t-exl3i-gpu.txt)"; fi
kld() {   # kld TAG PACK
  local t=$1 m=$2; [ -f $R/kld-$t.json ] && return 0; say "kld $t"
  ( flock -s 6
    ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$W/home timeout 1800 $W/install/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-$t.json.part > $R/kld-$t.log 2>&1 \
      && mv $R/kld-$t.json.part.rows $R/kld-$t.json.rows && mv $R/kld-$t.json.part $R/kld-$t.json
    say "kld $t rc $? $(sha256sum $R/kld-$t.json.rows 2>/dev/null | cut -c1-16)" ) 6>$L/ram.lock
}
kld V2-r10 $Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad
kld K4-r10 $V/packs/K4-r7.rad
say "VALBOX DONE"
