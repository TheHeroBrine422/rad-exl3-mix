#!/bin/bash
# mk17602.sh CMD ... -- #176-02 (w159) packs for the decode prices on ds8, on the K1 mad level's pack (heads: K1.125) (small bases;
# prices are per GEMV, so the base only shifts the fixed part). Outputs W/packs/*.rad (atomic: .part, then mv; a failed
# build leaves NAME.rad.fail so a waiting runner row moves on).
#   head PLAN      h-PLAN.rad: rad-convert --reuse of the K1 pack with its own recipe + the R16 head plan's line
#                  (u<K>: heads/q/R16s003mad/K<K> codebook=mad; u<K>m and u7: R16s003m1/K<K> codebook=mul1; rows=r16)
#   hd PLAN        hd-PLAN.rad: the same, kept (a held-out mix's head donor; radmix reads output.weight from it)
#   emb OPT        e-OPT.rad: radmix, token_embd.weight from emb/trellis (t<K>: emb-t-i8-K<K>; t<K>m: emb-t-m1-K<K>)
#   ab T4|T5       ab-T4.rad: radmix, the 48 GDN layers' ssm_a / ssm_b dropped and mad/ab/donor-ab-K<4|5>'s ssm_ab added
#   mix NAME FILE  m-NAME.rad: radmix of FILE (radmix lines; base = the K1 pack unless FILE's first line is "#base PATH")
#   roll N PLAN..  heads in order, at most N built-but-unmeasured head packs on disk at once; a head pack is deleted once
#                  its runner row is done (res/h-PLAN.jsonl)
#   a8 K           a8-K.rad: the uniform level K's pack + "act <w> a8g128" on the 336 trunk linears (recipes/a8.mix), for
#                  the int8-decode column (lead note 2:30 PM CST: QWEN35_EXL3_AQ_XB=1 EXL3I_MAD_GEMV=1 needs a8g128 weights)
#   gc             delete every h- / e- / ab- pack whose row is done (m- packs are kept until the residual rounds end)
# radmix = W/tools/radmix (w147's build of mix/radmix.cpp 243ebc26: add / drop lines; tools/radmix on ds8 predates them).
# Every build waits while W/QUIET exists and holds W/quiet.lock shared (run17602.sh QUIET=1 rows hold it exclusive).
# Locks: rad-convert holds conv.lock + ram.lock shared (w151's donor151.sh convention); radmix ram.lock shared. (w159)
set -u
Q=/data/q27; W=$Q/w159; P=$W/packs; R=$W/res; LK=$Q/locks; mkdir -p $P $W/recipes $W/logs
I=$Q/rad10-perf/install; HP=$Q/mad/rade05/home   # the packs-mad pack home (w151: e054e71 plugins, mad importer)
LV=q27v3-635bac02-k1-mad; B0=$Q/packs-mad/$LV-s6c57a3d1.rad
HLV=q27v3-635bac02-k1.125-mad; HB0=$Q/packs-mad/$HLV-s6c57a3d1.rad   # head packs: K1.125 (K1's final/ has moved to box2; the importer reads src: K1.125's final/ hard-linked to W/base-<level> so a trim cannot take it)
quiet() { while [ -f $W/QUIET ]; do sleep 20; done; }   # run17602.sh's e2e rounds: no build starts while they run
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S') [mk] $*" >> $W/logs/mk.log; }

head() {   # PLAN [PREFIX: h (measured, then gc'd) | hd (a held-out mix's donor, kept)]
  # w158's census-checked slim donor (hdonor173.sh: output.weight only) when it exists: radmix onto the K1.125 pack (no
  # rad-convert); returns 2 while it does not exist yet. HEAD_CONVERT=1: our own rad-convert (as for h-u4).
  local p=$1 x=${2:-h} o=$P/${2:-h}-$1.rad dn=$Q/w158/heads/hd-$1.rad
  [ -f $o ] && return 0; [ $x = h ] && [ -f $R/h-$p.jsonl ] && return 0
  if [ "${HEAD_CONVERT:-0}" != 1 ]; then
    [ -f $dn ] || return 2
    echo "output.weight $dn" > $W/recipes/$x-$p.mix
    radmix $o $W/recipes/$x-$p.mix $HB0; return
  fi
  local k cb d; k=${p#u}; k=${k%m}
  if [[ $p == *m || $p == u7 ]]; then cb=mul1; d=$Q/heads/q/R16s003m1/K$k; else cb=mad; d=$Q/heads/q/R16s003mad/K$k; fi
  [ -f $d/q-head.safetensors ] || { log "head $p: no $d/q-head.safetensors"; touch $o.fail; return 1; }
  { sed "s|src=$Q/ladder-mad/$HLV/final|src=$W/base-$HLV/final|" $W/base-$HLV/final.recipe; echo "output.weight exl3i src=$d codebook=$cb rate=$k rows=r16"; } > $W/recipes/$x-$p.recipe
  rm -f $o.part; quiet; local t=$(date +%s)
  if flock -s $W/quiet.lock flock $LK/conv.lock flock -s $LK/ram.lock env RADIANCE_HOME=$HP RADIANCE_HOST_DEVICES=1 nice -n 10 ionice -c3 \
       $I/bin/rad-convert --no-error $Q/models/qwen38-27b-bf16 --recipe $W/recipes/$x-$p.recipe \
       --set quantization_config.quant_method=exl3 --reuse $HB0 -o $o.part > $W/logs/$x-$p.convert.log 2>&1; then
    mv $o.part $o; log "head $p ($cb K$k): $(( $(date +%s) - t )) s, $(stat -c %s $o) B"
  else rm -f $o.part; touch $o.fail; log "head $p FAILED ($W/logs/h-$p.convert.log)"; return 1; fi
}

radmix() {  # OUT MIXFILE [BASE]
  local o=$1 mf=$2 b=${3:-$B0}
  [ -f $o ] && return 0
  rm -f $o.part; quiet
  if flock -s $W/quiet.lock flock -s $LK/ram.lock nice -n 10 ionice -c3 $W/tools/radmix $o.part $b $mf > $o.mixlog 2>&1; then
    mv $o.part $o; log "mix $(basename $o): $(stat -c %s $o) B ($(tail -n 1 $o.mixlog | cut -c1-150))"
  else rm -f $o.part; touch $o.fail; log "mix $(basename $o) FAILED ($o.mixlog)"; return 1; fi
}

emb() {
  local e=$1 k f; k=${e#t}; k=${k%m}
  if [[ $e == *m ]]; then f=$Q/emb/trellis/emb-t-m1-K$k.rad; else f=$Q/emb/trellis/emb-t-i8-K$k.rad; fi
  echo "token_embd.weight $f" > $W/recipes/e-$e.mix
  radmix $P/e-$e.rad $W/recipes/e-$e.mix
}

ab() {
  local t=$1 k=${1#T} l
  for l in $(seq 0 63); do
    [ $((l % 4)) = 3 ] && continue
    echo "drop blk.$l.ssm_a.weight"; echo "drop blk.$l.ssm_b.weight"; echo "add blk.$l.ssm_ab.weight $Q/mad/ab/donor-ab-K$k.rad"
  done > $W/recipes/ab-$t.mix
  radmix $P/ab-$t.rad $W/recipes/ab-$t.mix
}

gc() {
  local f t
  for f in $P/h-*.rad $P/e-*.rad $P/ab-*.rad; do
    [ -f $f ] || continue; t=$(basename $f .rad)
    [ -f $R/$t.jsonl ] && { rm -f $f; log "gc $t"; }
  done
}

cmd=$1; shift
case $cmd in
  head) head $1;;
  hd) head $1 hd;;
  emb) emb $1;;
  ab) ab $1;;
  mix) b=$(sed -n '1s/^#base //p' $2); radmix $P/m-$1.rad $2 ${b:-$B0};;
  gc) gc;;
  a8) radmix $P/a8-$1.rad $W/recipes/a8.mix $(awk -v k=$1 '$1 == k {print $2}' $W/recipes/uniform.txt);;   # uniform level + act a8g128 on the 336 trunk linears
  roll)
    n=$1; shift; pend=("$@")
    while [ ${#pend[@]} -gt 0 ]; do
      left=()
      for p in "${pend[@]}"; do
        while :; do
          gc; c=0
          for f in $P/h-*.rad; do [ -f $f ] && c=$((c + 1)); done
          [ $c -lt $n ] && break; sleep 30
        done
        head $p; rc=$?
        [ $rc = 2 ] && left+=($p)
      done
      pend=("${left[@]}")
      if [ ${#pend[@]} -gt 0 ]; then
        if ! pgrep -f "w158/code/hdonor173.sh" > /dev/null && ! pgrep -f "code/hdonor173.sh" > /dev/null; then
          log "roll: no w158 donor builder running; own rad-convert for ${pend[*]}"; export HEAD_CONVERT=1
        else sleep 120; fi
      fi
    done
    log "roll: all built"
    while ls $P/h-*.rad > /dev/null 2>&1; do sleep 60; gc; done
    log "roll: done";;
esac
