#!/bin/bash
# run4070.sh -- #100c quality arms on the 4070 (r100c.md "Rules"). Idempotent: an arm is skipped once its two score
# files exist; ourq resumes from its shards. One unit = quant + search eval + verify eval of one arm, under
# /data/yb/gpu.lock (one hold, < 15 min; d1112 interleaves); the arm's wq copy is deleted once scored.
# Steps (R100C_STEPS, default "check main side pairs report"):
#   check  hashcheck.py on the card (seconds): the hash build's values == the numpy map, every K used
#   main   i8 / hash at K3, K2, K4, seed offsets 0, 1      side  hiqr (IQR's A 0xB5297B) at K3, K2, K4, seed 0
#   pairs  pcmp.py pairs -> O/pairs.jsonl                  report  report.py -> O/report.md
# Stop: touch /data/yb/w48/STOP (checked between units). Needs setup4070.sh first.
P=/workspace/rad-exl3-mix/proxy2b; HERE=$(dirname "$(readlink -f "$0")")
W=/data/yb/w48; O=$W/r100c; M=/data/models/qwen35-08b; GS=/data/g08-search1; GV=/data/g08-verify1
SRC=/data/src/exllamav3/exllamav3/exllamav3_ext; D12=/data/yb/d1112
HO=(--hout /data/yb/c18/hdiag-snap1024-dy --sigma-reg-out 0.01)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
STEPS=${R100C_STEPS:-check main side pairs report}
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
stopped() { [ -f $W/STOP ]; }
want() { [[ " $STEPS " == *" $1 "* ]] && ! stopped; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; T0=$(date +%s); }
unlock() { flock -u 9; local s=$(( $(date +%s) - T0 )); [ $s -gt 840 ] && log "WARN: hold $s s > 14 min"; HOLD=$s; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
nq() { ls $1/*.safetensors 2>/dev/null | wc -l; }
declare -A HA=([hash]=0xd0474b [hiqr]=0xb5297b)
# ev TAG WQ GOLDEN: O/TAG.f32 (evalq writes .part -> rename); caller holds the lock
ev() { local tag=$1 wq=$2 g=$3 r; [ -f $O/$tag.f32 ] && return 0
  r=$(python3 $P/evalq.py $M $g --wq $wq --out $O/$tag.f32 2>> $O/eval.err | tail -n 1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')
  [ -f $O/$tag.f32 ] || { log "EVAL FAIL $tag"; return 1; }; log "$tag: $r"; }
# unit CB K SEED: quant (CB i8: the stock q2b.sh, as d1112's edy; hash / hiqr: q08h.sh) + both evals, one hold
unit() { local cb=$1 k=$2 s=$3; local t=$cb'k'$k$(sfx $s); local d=$O/q-$t
  [ -f $O/$t.f32 ] && [ -f $O/vf-$t.f32 ] && return 0
  stopped && return 1
  lock
  if [ $(nq $d/wq) -lt 150 ]; then
    if [ $cb = i8 ]; then $P/q2b.sh 08 $k i8 $d "${HO[@]}" --seed-offset $s > $d.log 2>&1
    else EXL3HASH_A=${HA[$cb]} bash $HERE/q08h.sh $k $d "${HO[@]}" --seed-offset $s > $d.log 2>&1; fi
  fi
  [ $(nq $d/wq) -ge 150 ] || { unlock; log "QUANT FAIL $t (see $d.log)"; return 1; }
  ev $t $d/wq $GS; ev vf-$t $d/wq $GV
  unlock; log "unit $t: hold $HOLD s"
  [ -f $O/$t.f32 ] && [ -f $O/vf-$t.f32 ] && rm -rf $d/wq; }
# pair A B GOLDEN [A.f32 path]: pcmp.py line into pairs.jsonl, once
pair() { local a=$1 b=$2 g=$3 fa=${4:-$O/$1.f32}
  grep -qF "\"pair\": \"$a -> $b\"" $O/pairs.jsonl 2>/dev/null && return 0
  [ -f $fa ] && [ -f $O/$b.f32 ] || return 1
  python3 $P/pcmp.py $g $fa $O/$b.f32 | tail -n 1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); d['pair']='$a -> $b'; print(json.dumps(d))" >> $O/pairs.jsonl; }

if want check && [ ! -f $O/check.ok ]; then
  mkdir -p /tmp/w48-run; lock; ( cd /tmp/w48-run
    PYTHONPATH=/data/src/exllamav3 TORCH_EXTENSIONS_DIR=$W/torch_ext TORCH_CUDA_ARCH_LIST=8.9 \
      python3 $W/code/v3/e3/exl3hash/hashcheck.py $SRC 2 3 4 5 2.5 ) > $O/check.log 2>&1; rc=$?; unlock
  log "hashcheck rc $rc: $(grep -c 'True' $O/check.log) / 5 K pass"; [ $rc = 0 ] && touch $O/check.ok || exit 1
fi
if want main; then for s in 0 1; do for k in 3 2 4; do for cb in i8 hash; do unit $cb $k $s; done; done; done; fi
if want side; then for k in 3 2 4; do unit hiqr $k 0; done; fi
if want pairs || want report; then
  for s in 0 1; do for k in 3 2 4; do x=$(sfx $s)
    pair i8k$k$x hashk$k$x $GS; pair vf-i8k$k$x vf-hashk$k$x $GV
    pair d1112:k${k}ste$x i8k$k$x $GS $D12/k${k}ste$x.f32; done; done
  for k in 3 2 4; do pair i8k$k hiqrk$k $GS; pair vf-i8k$k vf-hiqrk$k $GV; pair hashk$k hiqrk$k $GS; done
fi
want report && python3 $HERE/report.py $O > $O/report.md.part 2>> $O/report.err && mv $O/report.md.part $O/report.md
stopped && { log "stopped (STOP)"; exit 0; }
log "R100C RUN END"
