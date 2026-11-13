#!/bin/bash
# valr374.sh CARD -- #170 lead note 11-13 2:55: the 374503f build with media (/data/q27/rad10/r374: rad10build.sh with
# OUT / SRC / FF) against the a3799b5 rad10 build, on one card (card lock, ram.lock shared; the pack under conv.lock +
# ram.lock exclusive). Restartable. Log r374/val/val.log. Units:
#   1 tests: core ctest -L fast (FFmpeg now on: images_test / videos_test), exl3i -L gpu with test_twopass_fuse
#   2 device code: every kernel / architecture .so's .hip_fatbin in r374/home vs rad10/home (VOPD repeatable: unchanged
#     sources must give the same device code)
#   3 pack K4 (ladder k4 final, copied to r374/levels) by r374 and by rad10 (home-pack): bytes, v.* / dflash weights
#   4 KLD on golden-ver3-8k: V2 3.00, K4 (that pack); rows sha vs rad10's (89faf27a…, d7ee9f60…)
#   5 serve the K4 pack with media on: 3 image prompts (w124's img/, genimg.py), greedy text
#   6 bench_twopass A8 at K4, M 512 / 2048, EXL3I_A8_FUSE 0 / 1, twice (w130)
set -u; CARD=$1; W=/data/q27/rad10; O=$W/r374; Q=/data/q27; V=$O/val; R=$V/res; L=$Q/locks; G=$Q/goldens/golden-ver3-8k
PORT=$((8130 + CARD)); OBJ=/opt/rocm-10.1/lib/llvm/bin/llvm-objcopy; export PATH=$Q/venv/bin:$PATH
mkdir -p $R $V/packs; exec >> $V/val.log 2>&1; ulimit -n 65536
say() { echo "$(date -u +%T) $*"; }
exec 8>$L/card$CARD.lock; flock -x 8; say "card$CARD.lock held"
if [ ! -f $R/t-core-fast.txt ]; then ctest --test-dir $O/build -L fast -j8 --output-on-failure > $R/t-core-fast.txt.part 2>&1; mv $R/t-core-fast.txt.part $R/t-core-fast.txt; say "core fast: $(grep 'tests passed' $R/t-core-fast.txt)"; fi
if [ ! -f $R/t-exl3i-gpu.txt ]; then ROCR_VISIBLE_DEVICES=$CARD ctest --test-dir $O/pb/exl3i -L gpu --output-on-failure > $R/t-exl3i-gpu.txt.part 2>&1; mv $R/t-exl3i-gpu.txt.part $R/t-exl3i-gpu.txt; say "exl3i gpu (FUSE test included): $(grep 'tests passed' $R/t-exl3i-gpu.txt)"; fi
if [ ! -f $R/fatbin.txt ]; then
  T=$(mktemp -d); same=0; diff=0
  for f in $(cd $W/home && find kernels architectures quantizers -name '*.so' | sort); do
    [ -f $O/home/$f ] || continue
    $OBJ --dump-section=.hip_fatbin=$T/a $W/home/$f /dev/null 2>/dev/null || continue
    $OBJ --dump-section=.hip_fatbin=$T/b $O/home/$f /dev/null 2>/dev/null || continue
    if cmp -s $T/a $T/b; then same=$((same + 1)); else diff=$((diff + 1)); echo "differs: $f"; fi
  done > $R/fatbin.txt.part; echo "fatbin same $same differ $diff" >> $R/fatbin.txt.part; rm -rf $T
  mv $R/fatbin.txt.part $R/fatbin.txt; say "$(tail -n 1 $R/fatbin.txt)"
fi
# the ladder's k4 final (copied from ds8: box2's ladder dir lacks it) under r374/levels
sed "s|/data/q27/ladder/|$O/levels/|" $O/levels/q27v3-144dd6e1-k4-i8/final.recipe > $V/K4.recipe
# the recipe names its level's path and the container records the recipe, so bytes are compared on this path: K4 by
# r374 (rad-convert + home) vs K4p by rad10 (rad-convert + home-pack, whose packs equal the 7.14 ones byte for byte on
# ds8's ladder path)
pack() {   # pack NAME INSTALL HOME
  local n=$1 in=$2 h=$3; [ -f $V/packs/$n.rad ] && return 0
  ( flock -x 7; flock -x 6; say "pack $n"
    cd $Q && RADIANCE_HOME=$h RADIANCE_HOST_DEVICES=1 nice -n 10 $in/bin/rad-convert models/qwen38-27b-bf16 --recipe $V/K4.recipe \
      --set quantization_config.quant_method=exl3 --no-error -o $V/packs/$n.rad.part > $R/pack-$n.log 2>&1 && mv $V/packs/$n.rad.part $V/packs/$n.rad
    say "pack $n rc $? $(stat -c %s $V/packs/$n.rad 2>/dev/null) sha $(sha256sum $V/packs/$n.rad | cut -c1-16)"
  ) 7>$L/conv.lock 6>$L/ram.lock
  RADIANCE_HOME=$h $in/bin/rad-info $V/packs/$n.rad > $R/info-$n.txt 2>&1
  say "$n pack weights: $(grep -cE '^ *v\.' $R/info-$n.txt) v.*, $(grep -cE 'dflash' $R/info-$n.txt) dflash"
}
pack K4 $O/install $O/home; pack K4p $W/install $W/home-pack
kld() {   # kld TAG PACK
  local t=$1 m=$2; [ -f $R/kld-$t.json ] && return 0; say "kld $t"
  ( flock -s 6
    ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$O/home timeout 1800 $O/install/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-$t.json.part > $R/kld-$t.log 2>&1 \
      && mv $R/kld-$t.json.part.rows $R/kld-$t.json.rows && mv $R/kld-$t.json.part $R/kld-$t.json
    say "kld $t rc $? rows sha $(sha256sum $R/kld-$t.json.rows 2>/dev/null | cut -c1-16)" ) 6>$L/ram.lock
}
kld V2 $Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad
kld K4 $V/packs/K4.rad
say "rad10 rows: V2 89faf27ad364b71d, K4 d7ee9f60febe6647"
if [ ! -f $R/img.done ]; then
  ( flock -s 6; say "serve K4 with media"
    ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$O/home $O/install/bin/radiance --model $V/packs/K4.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 2 --mm-max-patches 4096 --host 127.0.0.1 --port $PORT > $R/serve-img.log 2>&1 &
    SP=$!; i=0
    while [ $i -lt 900 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 2; i=$((i + 2)); kill -0 $SP 2>/dev/null || break; done
    if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
      grep -E '"(hopper-describe|gull-what|dashboard-read)"' /data/q27/w124/prompts/img-prompts.jsonl > $V/img3.jsonl
      python3 -I /data/q27/w124/tools/genimg.py $PORT $V/img3.jsonl /data/q27/w124/img $R/img.jsonl 96 > $R/img.txt 2>&1
      touch $R/img.done; say "images: $(cat $R/img.txt | tr '\n' ' ')"
    else say "serve: no health ($(grep -m1 '^E ' $R/serve-img.log | cut -c1-200))"; fi
    kill $SP; wait $SP ) 6>$L/ram.lock
fi
if [ ! -f $R/fbench.txt ]; then
  B=$O/pb/exl3i; S=$(find $B -path '*radiance_home/*' -name exl3i.so | head -1)
  for r in 1 2; do for f in 0 1; do echo "== FUSE=$f round $r"
    EXL3I_A8_FUSE=$f ROCR_VISIBLE_DEVICES=$CARD timeout 600 $B/bench_twopass $S --rate 4 --m 512,2048 --modes a8 2>&1; echo "rc $?"
  done; done > $R/fbench.txt.part; mv $R/fbench.txt.part $R/fbench.txt; say "fbench written"
fi
say "VALR374 DONE"
