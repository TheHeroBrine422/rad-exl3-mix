#!/bin/bash
# val10.sh -- #170: the ROCm 10.1 radiance build (W/install + W/home, rad10build.sh) against the 7.14 one on one card
# (CARD, default 6; card lock, ram.lock shared; packs under conv.lock + ram.lock exclusive, --no-error). The 7.14 side is
# the same source: the frozen core /data/q27/rad/install (42960d5; core files = a3799b5's) + W/ref714/home (plug714.sh).
# Restartable: every unit skips when its output exists. Log W/val/val.log, results W/val/res. Units, in order:
#   1 tests: core ctest -L fast; core ctest -L gpu; each plugin's -L gpu (exl3i's test_twopass_fuse kept for the end)
#   2 packs (rad-convert with the pack homes of packhome.sh; PACKONLY=1 stops after them): K4 (the ladder's k4 final.recipe; sha256 vs the 7.14 pack in /data/q27/packs), K4H (K4 + the R16 K7 mul1 head,
#     w124's line) by both builds, K4A (k4.125 + act a8g128 on the 336 trunk weights, w124's wm5) by 10.1
#   3 KLD on golden-ver3-8k: V2 3.00 (frozen 7.14 home, ref714, 10.1), K4, K4H (each pack by its own build + 10.1 on the
#     7.14 pack), K4A FUSE off (7.14, 10.1)
#   4 serve K4H: text sanity (w124's ask.sh) and decode step C 1 (w124's dstep.sh: short, 8k), 7.14 / 10.1 interleaved x 2
#   5 FUSE: K4A KLD with EXL3I_A8_FUSE=1 on 10.1 (hung the R9700 built with 7.14), then test_twopass_fuse (timeouts)
set -u
W=/data/q27/rad10; Q=/data/q27; V=$W/val; CARD=${CARD:-6}; L=$Q/locks; PORT=$((8130 + CARD))
I10=$W/install; H10=$W/home; I7=$Q/rad/install; H7=$W/ref714/home; HF=$Q/rad/home
G=$Q/goldens/golden-ver3-8k; LV=$Q/ladder; HH=$Q/heads/q; T=$Q/w124/tools; PR=$Q/w124/prompts
P=$V/packs; R=$V/res; PY=$Q/venv/bin/python; export PATH=$Q/venv/bin:$PATH   # ctest
mkdir -p $P $R $V/recipes; exec >> $V/val.log 2>&1; ulimit -n 65536
say() { echo "$(date -u +%T) $*"; }
say "start (pid $$, card $CARD)"
exec 8>$L/card$CARD.lock; flock -x 8; say "card$CARD.lock held"

# 1 unit tests (SKIPCORE=1: not the core suites, e.g. while another card runs them)
[ -n "${SKIPCORE:-}" ] || if [ ! -f $R/t-core-fast.txt ]; then ctest --test-dir $W/build -L fast -j8 --output-on-failure > $R/t-core-fast.txt.part 2>&1; mv $R/t-core-fast.txt.part $R/t-core-fast.txt; say "core fast: $(tail -n 3 $R/t-core-fast.txt | head -n 1)"; fi
[ -n "${SKIPCORE:-}" ] || if [ ! -f $R/t-core-gpu.txt ]; then ROCR_VISIBLE_DEVICES=$CARD ctest --test-dir $W/build -L gpu --output-on-failure --timeout 1800 > $R/t-core-gpu.txt.part 2>&1; mv $R/t-core-gpu.txt.part $R/t-core-gpu.txt; say "core gpu: $(grep 'tests passed' $R/t-core-gpu.txt)"; fi
for p in exl3quant qwen35_exl3 exl3dense exl3r4d exl3i; do
  [ -f $R/t-$p-gpu.txt ] && continue
  ROCR_VISIBLE_DEVICES=$CARD ctest --test-dir $W/pb/$p -L gpu -E test_twopass_fuse --output-on-failure --timeout 900 > $R/t-$p-gpu.txt.part 2>&1
  mv $R/t-$p-gpu.txt.part $R/t-$p-gpu.txt; say "$p gpu: $(grep -E 'tests passed|No tests' $R/t-$p-gpu.txt)"
done

# 2 packs
cp $LV/q27v3-144dd6e1-k4-i8/final.recipe $V/recipes/K4.recipe
{ cat $V/recipes/K4.recipe; echo "output.weight exl3i src=$HH/R16s003m1/K7 codebook=mul1 rows=r16"; } > $V/recipes/K4H.recipe
cp $LV/q27v3-144dd6e1-k4.125-i8/final.recipe $V/recipes/K4A.recipe
ACT=(); for l in $(seq 0 63); do
  if [ $((l % 4)) = 3 ]; then ws="attn_qg attn_k attn_v attn_output"; else ws="ssm_qkv ssm_z ssm_out"; fi
  for w in $ws ffn_gate_up ffn_down; do ACT+=(--set "exl3.act.blk.$l.$w.weight=a8g128"); done
done
pack() {   # pack NAME RECIPE INSTALL HOME [extra]
  local n=$1 rc=$2 in=$3 h=$4; shift 4
  [ -f $P/$n.rad ] && return 0
  rm -f $P/$n.rad.part; say "pack $n: waiting for conv.lock + ram.lock"
  ( flock -x 7; flock -x 6; say "pack $n: start"
    cd $Q && RADIANCE_HOME=$h RADIANCE_HOST_DEVICES=1 nice -n 10 $in/bin/rad-convert models/qwen38-27b-bf16 --recipe $V/recipes/$rc.recipe \
      --set quantization_config.quant_method=exl3 "$@" --no-error -o $P/$n.rad.part > $R/pack-$n.log 2>&1 && mv $P/$n.rad.part $P/$n.rad
    say "pack $n: rc $? $(stat -c %s $P/$n.rad 2>/dev/null)"
  ) 7>$L/conv.lock 6>$L/ram.lock
  [ -f $P/$n.rad ] && sha256sum $P/$n.rad | cut -c1-64 > $P/$n.sha
}
K7P=$Q/packs/q27v3-144dd6e1-k4-i8-s6c57a3d1.rad
# pack homes (packhome.sh): a3799b5's qwen35_exl3 cannot pack from the HF checkpoint; 1aa18de's in its place
HP10=$W/home-pack; HP7=$W/ref714/home-pack
pack K4-r10 K4 $I10 $HP10
[ -f $P/K4-r7.sha ] || { [ -f $K7P.sha ] || sha256sum $K7P | cut -c1-64 > $P/K4-r7.sha; }
say "K4 pack sha: 10.1 $(cat $P/K4-r10.sha) 7.14 $(cat $P/K4-r7.sha)"
pack K4H-r10 K4H $I10 $HP10; pack K4H-r7 K4H $I7 $HP7
say "K4H pack sha: 10.1 $(cat $P/K4H-r10.sha) 7.14 $(cat $P/K4H-r7.sha)"
pack K4A-r10 K4A $I10 $HP10 "${ACT[@]}"
[ -n "${PACKONLY:-}" ] && { say "PACKS DONE"; exit 0; }

# 3 KLD
kld() {   # kld TAG INSTALL HOME PACK [env...]
  local t=$1 in=$2 h=$3 m=$4; shift 4
  [ -f $R/kld-$t.json ] && return 0
  [ -f $m ] || { say "kld $t: no pack $m"; return 0; }
  say "kld $t"
  ( flock -s 6
    env "$@" ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$h timeout ${KT:-1800} $in/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-$t.json.part > $R/kld-$t.log 2>&1 \
      && mv $R/kld-$t.json.part.rows $R/kld-$t.json.rows && mv $R/kld-$t.json.part $R/kld-$t.json
    say "kld $t rc $? $($PY -c "import json; j=json.load(open('$R/kld-$t.json'))['all']; print(j['kld']['mean'], j['top1_agreement'])" 2>/dev/null)"
  ) 6>$L/ram.lock
}
V2=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad
kld V2-r7f $I7 $HF $V2; kld V2-r7 $I7 $H7 $V2; kld V2-r10 $I10 $H10 $V2
kld K4-r7 $I7 $H7 $K7P; kld K4-r10 $I10 $H10 $K7P
cmp -s $P/K4-r10.sha $P/K4-r7.sha || kld K4-r10p $I10 $H10 $P/K4-r10.rad
kld K4H-r7 $I7 $H7 $P/K4H-r7.rad; kld K4H-r10 $I10 $H10 $P/K4H-r10.rad
cmp -s $P/K4H-r10.sha $P/K4H-r7.sha || kld K4H-r10x $I10 $H10 $P/K4H-r7.rad
kld K4A-r7-f0 $I7 $H7 $P/K4A-r10.rad EXL3I_A8_FUSE=0; kld K4A-r10-f0 $I10 $H10 $P/K4A-r10.rad EXL3I_A8_FUSE=0
$PY -I $W/scripts/kldcmp.py $R > $R/kldcmp.txt 2>&1; say "kldcmp written"

# 4 serve K4H: text + decode step
up() {   # up TAG INSTALL HOME PACK
  local t=$1 in=$2 h=$3 m=$4 i=0
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$h $in/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 12288 \
    --max-num-seqs 8 --host 127.0.0.1 --port $PORT > $R/serve-$t.log 2>&1 &
  SP=$!
  while [ $i -lt 900 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 2; i=$((i + 2)); kill -0 $SP 2>/dev/null || break; done
  curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null
}
down() { kill $SP 2>/dev/null; wait $SP 2>/dev/null; }
serve() {   # serve TAG INSTALL HOME PACK
  local t=$1; [ -f $R/s-$t.done ] && return 0
  [ -f $4 ] || { say "serve $t: no pack"; return 0; }
  ( flock -s 6; say "serve $t"
    if up "$@"; then
      [ "${t##*-}" = 1 ] && bash $T/ask.sh $PORT ctx8k > $R/ask-$t.txt 2>&1
      P=$PORT PROMPT=$PR/short.txt LEVELS=1 bash $T/dstep.sh > $R/dstep-$t-short.txt 2>&1
      P=$PORT PROMPT=$PR/ctx8k.txt LEVELS=1 bash $T/dstep.sh > $R/dstep-$t-8k.txt 2>&1
      touch $R/s-$t.done
    else say "serve $t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-160))"; fi
    down ) 6>$L/ram.lock
}
for r in 1 2; do serve K4H-r7-$r $I7 $H7 $P/K4H-r7.rad; serve K4H-r10-$r $I10 $H10 $P/K4H-r10.rad; done

# 5 FUSE re-test (last: a hang may leave the card unusable)
KT=900 kld K4A-r10-f1 $I10 $H10 $P/K4A-r10.rad EXL3I_A8_FUSE=1
if [ ! -f $R/t-fuse.txt ]; then ROCR_VISIBLE_DEVICES=$CARD timeout 900 ctest --test-dir $W/pb/exl3i -R test_twopass_fuse --output-on-failure > $R/t-fuse.txt.part 2>&1; echo "rc $?" >> $R/t-fuse.txt.part; mv $R/t-fuse.txt.part $R/t-fuse.txt; say "fuse test: $(grep -E 'tests passed|rc ' $R/t-fuse.txt | tr '\n' ' ')"; fi
$PY -I $W/scripts/kldcmp.py $R > $R/kldcmp.txt 2>&1
say "VAL DONE"
