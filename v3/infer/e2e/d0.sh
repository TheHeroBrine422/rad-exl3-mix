#!/bin/bash
# d0.sh [CARD] [LEVEL ...] -- EXL3-V3's first end-to-end check (D0): a ladder level, imported through exl3quant's exl3i
# quantiser into a .rad pack, its whole trunk swapped into V2 3.00 (radmix; V2's head and embedding kept), served by stock
# radiance 1.15.1 with our plugins (exl3i_dense on the exl3i kernel library) and scored on golden-ver1 against V2 3.00 on
# the same build (/data/port115, 0.027122). LEVEL: K values under /data/v3ladder2 (default 3 2.5). Out /data/v3e2e/.
#
# Before running: sync the fork (plugin/exl3i, exl3import, exl3quant, qwen35_exl3, exl3r4d, exl3dense) into /root/rhv3, and
# this script with eval/cmp115.py and store/census.py into /root/v3e2e/.
#
# PASS RULE (written 2026-11-02 before any run). Per level:
#   1. the pack converts and the mix loads (rc 0), the bpw is what the level says (trunk K, V2's head / embedding);
#   2. its gen-masked mean KLD on ver1 is finite and within [0.5x, 2x] of the knap4 expectation for a uniform trunk:
#      K3 0.02-0.06, K2.5 0.04-0.12. A wiring fault (record order, rate, codebook, scales) reads >= 0.5.
#   Rule 2 is a sanity band, not a quality claim: ver1 is a verify-pool slice, used once here; quality comes from D3.
: ${RXM_WORK:=/root}
C=${1:-1}; shift 2>/dev/null; LEVELS=${*:-3 2.5}
O=/data/v3e2e L=/data/v3ladder2 R=/root/rhv3 I=/root/install-115 G=$RXM_WORK/golden-ver1
V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad REF=/data/port115/r115.json CMP=${CMP:-/root/v3e2e/cmp115.py} CENSUS=${CENSUS:-/root/v3e2e/census.py}
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/result.txt; }
export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1

# 1. plugins: exl3quant (exl3i importer), exl3i (kernel), qwen35_exl3 (routes exl3i weights), exl3r4d, exl3dense
( exec 7>$RXM_WORK/ram.lock; flock -s 7
  for p in exl3quant exl3i qwen35_exl3; do
    { cmake -S $R/plugin/$p -B $R/build-$p -G Ninja -DCMAKE_PREFIX_PATH=$I -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_ARCHITECTURES=gfx1201 && cmake --build $R/build-$p -j6; } >> $O/build.log 2>&1 || { echo "BUILD $p FAIL"; exit 1; }
  done ) || { log "build FAIL ($O/build.log)"; exit 1; }
$R/build-qwen35_exl3/qwen35_exl3_test > $O/q35test.log 2>&1 && $R/build-exl3quant/exl3quant_test \
  $R/build-exl3quant/radiance_home/quantizers/exl3quant.so >> $O/q35test.log 2>&1 || { log "host tests FAIL ($O/q35test.log)"; exit 1; }
H=$O/home; rm -rf $H; cp -a $I/share/radiance $H
cp $R/build-exl3dense/radiance_home/kernels/exl3dense.so $R/build-exl3r4d/radiance_home/kernels/exl3_r4d.so \
   $R/build-exl3i/radiance_home/kernels/exl3i.so $H/kernels/
cp $R/build-exl3quant/radiance_home/quantizers/exl3quant.so $H/quantizers/
cp $R/build-qwen35_exl3/radiance_home/architectures/qwen35_exl3.so $H/architectures/
log "plugins built; host tests pass"

# 2. V2 3.00 (deleted in the 2026-11-02 cleanup; public, from our own HF repo)
[ -f $V2 ] || { mkdir -p /data/release; /root/hfvenv/bin/python -c "from huggingface_hub import hf_hub_download as d; \
  d('Hero422/qwen3.8-27b-exl3-v2-rad', '$(basename $V2)', local_dir='/data/release')" > $O/dl.log 2>&1 || { log "V2 download FAIL"; exit 1; }; }

TRUNK() {   # every trunk weight of the 64 layers (full attention every 4th layer, GDN the rest)
  for l in $(seq 0 63); do
    if [ $((l % 4)) = 3 ]; then for t in attn_qg attn_k attn_v attn_output; do echo blk.$l.$t.weight; done
    else for t in ssm_qkv ssm_z ssm_out; do echo blk.$l.$t.weight; done; fi
    echo blk.$l.ffn_gate_up.weight; echo blk.$l.ffn_down.weight
  done
}
for k in $LEVELS; do
  P=$O/pack-K$k.rad M=$O/v2base-trunkK$k.rad
  # 3. the pack: the level's trunk through the exl3i importer; everything else the checkpoint's own (bf16)
  if [ ! -f $P ]; then
    TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u | sed "s|\$| exl3i src=$L/K$k codebook=mul1i8 rate=$k|" > $O/K$k.recipe
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd /workspace; s=$(date +%s)
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $I/bin/rad-convert models/qwen38-27b-bf16 --recipe $O/K$k.recipe \
        --set quantization_config.quant_method=exl3 -o $P.part > $O/conv-K$k.log 2>&1 && mv $P.part $P
      echo "pack K$k rc=$? $(( $(date +%s) - s ))s" ) | tee -a $O/result.txt
    [ -f $P ] || { log "pack K$k FAIL ($O/conv-K$k.log)"; continue; }
  fi
  python3 -I $CENSUS $P --recipe $O/K$k.recipe > $O/census-pack-K$k.txt || { log "pack K$k census FAIL ($O/census-pack-K$k.txt)"; continue; }
  # 4. the mix: V2 3.00 with the whole trunk from the pack
  TRUNK | sed "s|\$| $P|" > $O/K$k.mix
  /root/hproxy/radmix $M $V2 $O/K$k.mix > $O/mix-K$k.log 2>&1 || { log "radmix K$k FAIL"; continue; }
  python3 -I $CENSUS $M --mix $O/K$k.mix --base $V2 > $O/census-K$k.txt || { log "mix K$k census FAIL ($O/census-K$k.txt)"; continue; }
  grep -q "exl3i  *$k  *mul1i8 *f16  *336 " $O/census-K$k.txt || { log "mix K$k: not 336 exl3i K$k trunk weights ($O/census-K$k.txt)"; continue; }
  log "K$k: $(basename $M) $(python3 -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
  # 5. score (card lock, then ram, as everything on the cards)
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $O/K$k.json > $O/K$k.log 2>&1
    echo "score K$k rc=$? $(( $(date +%s) - s ))s" ) | tee -a $O/result.txt
  A=$(ls $REF.rows 2>/dev/null || ls ${REF%.json}.rows 2>/dev/null); B=$(ls $O/K$k.json.rows $O/K$k.rows 2>/dev/null | head -1)
  log "K$k vs V2 3.00 (ver1, gen-masked): $(python3 $CMP $G $A $B 2>&1 | tail -2 | tr '\n' ' ')"
done
