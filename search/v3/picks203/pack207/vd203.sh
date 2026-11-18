#!/bin/bash
# vd203.sh base | SIZE... (box2; w203, 11-18) -- the V3 release containers with the bf16 vision tower and the z-lab DFlash2
# drafter in them, as IQR-V4's files carry them. Restartable: finished units are skipped; logs in vramvd/logs/, vramvd/work/.
#
#  base    base-T4vd.rad, the BASE of every container from here on (v3/df2 #180 rule: the base carries the drafter lines and
#          --draft-model, so every radmix over it inherits the tower, the drafter and their metadata):
#          (1) rad-convert of the bf16 checkpoint with the drafter lines (dflash-lines.recipe, no rerank_head line: the drafter
#              is rescored against the trunk's own exl3i head, #180-02) and --draft-model, every trunk / head / embedding /
#              norm weight deferred (--defer), so the file holds v.* (tower), dflash.* (drafter), the draft.* / vision metadata,
#              and base-T4's metadata keys (quantization_config, the 48 exl3.act ssm_ab keys);
#          (2) radmix over (1) with an `add` line for every base-T4 weight (a byte copy from base-T4.rad, 739 weights).
#              Check (whash): every base-T4 weight byte-identical in base-T4vd; the only extra weights are v.* and dflash.*.
#          (base-T4's trunk / ssm_ab sources, ladder-mad K1 and absrc-K4, are not on box2: the convert never needs them.)
#  SIZE    radmix -> vramvd/SIZE.rad.part over base-T4vd.rad with work/SIZE/release.mix, the same mix as pack207's, where the
#          trunk donor work/SIZE/trunk.rad (deleted by pack207) is vram/SIZE.rad (same bytes: its 336 trunk weights; the head,
#          embedding and act lines unchanged). Census as pack207.sh (no --expect-emb) -> vramvd/SIZE.rad on PASS.
#  verify  final750 exactly as rel207.sh (card $VCARD, ram shared, golden-final750, --kv-cache-dtype bf16, max-model-len 16448);
#          rows must be byte-identical to verify/SIZE-final750.json.rows (else mad205/scores/SIZE-mad-final750.json.rows)
#          -> state/SIZE.vd-ok or state/SIZE.vd-bad. Outputs: vramvd/verify/.
# Locks: conv.lock + ram.lock shared around rad-convert / radmix (as pack207), card$VCARD.lock + ram shared around radiance.
set -u
Q=/data/q27; NEW=$Q/w201/rad21; MODEL=$Q/models/qwen38-27b-bf16; DF=$Q/models/qwen38-27b-dflash2
RADMIX=$Q/w158/tools/radmix; CEN=$Q/w158/code/census.py; INFO=$NEW/install/bin/rad-info
R=$Q/release-v3; V=$R/vramvd; W0=$V/work; B0=$Q/srch183/base/base-T4.rad; BVD=$R/base-T4vd.rad
G=$Q/final/golden-final750; VCARD=${VCARD:-0}; LK=$Q/locks; PY="/data/q27/venv/bin/python -I"
export RADIANCE_HOME=$NEW/home RADIANCE_HOST_DEVICES=1
mkdir -p $V/logs $V/verify $W0 $R/state
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a $V/logs/vd203.log; }
room() { [ "$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)" -ge 100 ]; }

base() {
  [ -f $BVD ] && { say "base: have $BVD"; return 0; }
  [ -f $DF/model.safetensors ] || { say "base: no $DF/model.safetensors"; return 1; }
  room || { say "base: under 100 GB free on /data, stop"; return 1; }
  local W=$W0/base M=$W0/base/meta.rad t0; mkdir -p $W
  # the 48 exl3.act keys of base-T4 (its ssm_ab weights' activation), as --set: metadata, seen by declare and written
  local A=() D=() k v
  while read -r k v; do A+=(--set "$k=$v"); done < <($INFO --meta $B0 | awk '$1 ~ /^exl3\.act\.blk\..*\.ssm_ab\.weight$/ {print $1, $2}')
  for n in 'blk.*' 'output.weight' 'token_embd.weight' 'output_norm.weight'; do D+=(--defer "$n"); done
  say "base: $(( ${#A[@]} / 2 )) exl3.act key(s) as --set, ${#D[@]} --defer arg(s)"
  if [ ! -f $M ]; then
    rm -f $M.part; t0=$(date +%s)
    ( cd $Q && flock $LK/conv.lock flock -s $LK/ram.lock nice -n 10 $NEW/install/bin/rad-convert --no-error $MODEL \
        --recipe $W0/dflash-lines.recipe --draft-model $DF "${D[@]}" --set quantization_config.quant_method=exl3 "${A[@]}" \
        -o $M.part ) > $W/convert.log 2>&1 || { say "base: FAIL convert ($W/convert.log)"; return 1; }
    mv $M.part $M; say "base: convert $(( $(date +%s) - t0 )) s, $(stat -c %s $M) bytes"
  fi
  $PY $W0/names.py $INFO $B0 > $W/names-base.txt && $PY $W0/names.py $INFO $M > $W/names-meta.txt || { say "base: names FAIL"; return 1; }
  # the mix: an add line per base-T4 weight that the metadata container does not hold (none of its weights are dropped)
  LC_ALL=C comm -23 <(LC_ALL=C sort $W/names-base.txt) <(LC_ALL=C sort $W/names-meta.txt) | awk -v B="$B0" '{print "add " $0 " " B}' > $W/base.mix
  say "base: $(wc -l < $W/base.mix) add line(s) from base-T4; meta holds $(wc -l < $W/names-meta.txt) weight(s)"
  rm -f $BVD.part; t0=$(date +%s)
  flock $LK/conv.lock flock -s $LK/ram.lock nice -n 10 $RADMIX $BVD.part $M $W/base.mix > $W/radmix.log 2>&1 \
    || { say "base: FAIL radmix ($W/radmix.log)"; return 1; }
  say "base: radmix $(( $(date +%s) - t0 )) s: $(tail -1 $W/radmix.log)"
  # whash: every base-T4 weight byte-identical; the extra weights are the tower and the drafter only
  $PY $W0/whash.py $INFO $B0 $BVD.part > $W/whash.txt 2>&1 || { say "base: whash FAIL ($W/whash.txt)"; return 1; }
  local both=$(grep '^common' $W/whash.txt) bad=$(grep -c '^  DIFF' $W/whash.txt) lonly=$(grep -c '^  ONLY A' $W/whash.txt)
  local extra=$(grep -c '^  ONLY B' $W/whash.txt) extra_bad=$(grep '^  ONLY B' $W/whash.txt | grep -vc ' v\.\| dflash\.')
  say "base: whash: $both; DIFF $bad, only-A $lonly, extra $extra (not v./dflash.: $extra_bad)"
  [ "$bad" = 0 ] && [ "$lonly" = 0 ] && [ "$extra_bad" = 0 ] || { say "base: CHECK FAILED, base-T4vd.rad.part kept"; return 1; }
  mv $BVD.part $BVD; say "base: DONE $BVD $(stat -c %s $BVD) bytes"
}

size() {
  local S=$1 O=$V/$S.rad W=$V/work/$S; mkdir -p $W
  [ -f $O ] && { say "$S: have $O"; return 0; }
  [ -f $R/vram/$S.rad ] || { say "$S: no vram/$S.rad yet"; return 1; }
  [ -f $R/work/$S/release.mix ] || { say "$S: no work/$S/release.mix"; return 1; }
  [ -f $BVD ] || { say "$S: no base"; return 1; }
  room || { say "STOP: under 100 GB free on /data before $S"; exit 1; }
  local MIX=$W/release-vd.mix n
  sed "s#^\(\S*\) $R/work/$S/trunk.rad\$#\1 $R/vram/$S.rad#" $R/work/$S/release.mix > $MIX
  n=$(grep -c " $R/vram/$S.rad\$" $MIX)
  [ "$n" = 336 ] || { say "$S: $n trunk donors in the mix, want 336"; return 1; }
  rm -f $O.part; local t0=$(date +%s)
  flock $LK/conv.lock flock -s $LK/ram.lock nice -n 10 $RADMIX $O.part $BVD $MIX > $W/radmix.log 2>&1 \
    || { say "$S: FAIL radmix ($W/radmix.log)"; return 1; }
  if $PY $CEN $O.part --mix $MIX --base $BVD --expect-act a8g128 --max-version 4 > $W/census.txt 2>&1; then
    mv $O.part $O; say "$S: $O $(stat -c %s $O) bytes, census PASS ($(( $(date +%s) - t0 )) s)"
  else say "$S: CENSUS FAIL ($W/census.txt), kept $O.part"; return 1; fi
}

verify() {
  local S=$1 O=$V/$S.rad o=$V/verify/$S-final750
  [ -f $R/state/$S.vd-ok ] || [ -f $R/state/$S.vd-bad ] && { say "$S: verify marked already"; return 0; }
  rm -f $o.json.part $o.json.part.rows
  flock $LK/card$VCARD.lock flock -s $LK/ram.lock env ROCR_VISIBLE_DEVICES=$VCARD RADIANCE_HOME=$NEW/home $NEW/install/bin/radiance \
    --model $O --tp 1 --kv-cache-dtype bf16 --max-model-len 16448 --max-num-seqs 1 --host-headroom-mib 8192 \
    --kld-ref $G --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
  local ref=$R/verify/$S-final750.json.rows; [ -f $ref ] || ref=$Q/mad205/scores/$S-mad-final750.json.rows
  if [ -f $o.json.rows ] && cmp -s $o.json.rows $ref; then
    touch $R/state/$S.vd-ok; say "verify $S: rows byte-identical to $ref ($(python3 -c "import json;print(json.load(open('$o.json'))['all']['kld']['mean'])" 2>/dev/null))"
  else touch $R/state/$S.vd-bad; say "verify $S: ROWS DIFFER or no rows (vs $ref; log $o.log)"; fi
}

if [ "${1:-}" = base ]; then shift; base || exit 1; fi
for S in "$@"; do size $S && verify $S; done
