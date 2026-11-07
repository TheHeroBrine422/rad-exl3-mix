#!/bin/bash
# g64.sh -- w64 items 1 (#111 writers asym under knapsack mixes) and 3 (#112 C9 alpha on the writers after asym), 0.8B,
# 4070 (../w64.md). Idempotent (rerun = continue: every product is skipped when its file / .done exists); every GPU unit
# holds /data/yb/gpu.lock (< 15 min each). Runs a snapshot: /data/yb/w64/snap/ (copied by the launcher).
# Steps (G64_STEPS, default "gate i1 i3s i3c report"):
#   gate  seed 0: fresh K3 edy + swj siblings with d1112's decoded k3st / k3ste writers vs d1112's k3st / k3ste_swj
#         scores (+-1%); shard determinism and deq precision reported. A gate fail stops everything.
#   i1    item 1, seeds 0-2: siblings K2.5 / K3.5, cells A (Kw 3, Ks 2.5), B (3, 3.5), C (2, 2.5); arms xc / xs / xm
#   i3s   item 3 screen, seed 0: alpha per writer family on fit, K3 / K2, stacks st (asym) and pl (plain)
#   i3c   item 3 confirm: the picked alpha* on seeds 0-2, search + verify
#   report  r64.py -> O/report.md
# Stop: touch /data/yb/w64/STOP (checked between units). Read-only inputs: /data/yb/d1112 (shards, scores).
P=/workspace/rad-exl3-mix/proxy2b X2=$P/exp/w02-fixedscale D7=$P/exp/w07-denoise
SNAP=$(dirname "$(readlink -f "$0")")
DEQ=$SNAP/deq.py C9W=$SNAP/c9w.py R64=$SNAP/r64.py
M=/data/models/qwen35-08b F=/data/g08-fit1 H=/data/h08/calib META=/data/challenge/calib-meta
O=${G64_OUT:-/data/yb/w64} D12=/data/yb/d1112
C18=/data/yb/c18/hdiag-snap1024-dy ONLY=/data/yb/w07-seqsib/only.txt
GS=/data/g08-search1 GV=/data/g08-verify1
HO=(--hout $C18 --sigma-reg-out 0.01)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
STEPS=${G64_STEPS:-gate i1 i3s i3c report}
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
stopped() { [ -f $O/STOP ]; }
want() { [[ " $STEPS " == *" $1 "* ]] && ! stopped; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; echo "$(date -u +%T) lock $1" >> $O/lock.log; }
unlock() { echo "$(date -u +%T) unlock" >> $O/lock.log; flock -u 9; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
nq() { ls $1/*.safetensors 2>/dev/null | wc -l; }
ev() { local tag=$1 wq=$2 g=$3 r; [ -f $O/$tag.f32 ] && return 0
  r=$(python3 $P/evalq.py $M $g --wq $wq --out $O/$tag.f32 2>> $O/eval.err | tail -n 1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')
  [ -f $O/$tag.f32 ] || { log "EVAL FAIL $tag"; return 1; }; log "$tag: $r"; }
pair() { local a=$1 b=$2 g=$3 fa=${4:-$O/$1.f32} fb=${5:-$O/$2.f32}
  grep -qF "pair $a -> $b: \"ratio\"" $O/result.txt 2>/dev/null && return 0
  [ -f $fa ] && [ -f $fb ] || return 1
  log "pair $a -> $b: $(python3 $P/pcmp.py $g $fa $fb | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
q2b() { local d=$1; shift; $P/q2b.sh 08 "$@" > $d.log 2>&1; }
# sib K S: edy trunk quant + swj at K, seed offset S (d1112 chain steps 1-2) -> O/q-k{K}e{x}_swj/wq (+ .done)
sib() { local k=$1 s=$2 x=$(sfx $2); local qd=$O/q-k$k$x sw=$O/q-k$k${x}_swj
  [ -f $sw/.done ] && return 0; stopped && return 1
  if [ $(nq $qd/wq) -lt 150 ]; then lock "quant k$k$x"; q2b $qd $k i8 $qd "${HO[@]}" --seed-offset $s; unlock
    [ $(nq $qd/wq) -ge 150 ] || { log "QUANT FAIL k$k$x"; return 1; }; fi
  mkdir -p $O/q-k$k${x}src; ln -sfn $qd/wq $O/q-k$k${x}src/wq; rm -rf $sw $O/q-k$k${x}_swu
  lock "swj k$k$x"; python3 $X2/swiglufit.py $M $O/q-k$k${x}src $F $sw $O/q-k$k${x}_swu > $sw.log 2>&1; unlock
  [ -f $sw/.done ] && [ $(nq $sw/wq) = 150 ] || { log "SWIGLUFIT FAIL k$k$x"; return 1; }
  log "sib k$k$x ready"; }
# momdn K S: sib-mode moments through the K siblings (qad, 1M tokens, seed 0) and G, F (lambda 1, rho 0)
momdn() { local k=$1 s=$2 x=$(sfx $2); local sw=$O/q-k$k${x}_swj mo=$O/mom-k$k$x dn=$O/dn-k$k$x
  [ -f $dn.done ] && return 0; stopped && return 1; sib $k $s || return 1
  if [ ! -f $mo.done ]; then lock "moments k$k$x"
    for r in 0-11 12-23; do python3 $D7/moments.py $M $META $mo --layers $r --tokens 1000000 --mode sib --wq $sw/wq --seed 0 >> $mo.log 2>&1 \
      || { unlock; log "MOMENTS FAIL k$k$x $r"; return 1; }; done
    unlock; touch $mo.done; log "moments k$k$x: $(grep '^MOMENTS DONE' $mo.log | tail -n 1)"; fi
  lock "mkdenoise k$k$x"; python3 $D7/mkdenoise.py $H $mo $dn --lam 1 --rho 0 --ckpt $M > $dn.log 2>&1; local rc=$?; unlock
  [ $rc = 0 ] || { log "MKDENOISE FAIL k$k$x"; return 1; }; touch $dn.done; rm -rf $mo; }
# wasym KW KS S: the 48 writers at KW requantized against the KS siblings' G, F -> shards O/q-w{KW}m{KS}{x}
wasym() { local kw=$1 ks=$2 s=$3 x=$(sfx $3); local wd=$O/q-w${kw}m$ks$x dn=$O/dn-k$ks$x
  [ $(ls $wd/q-L*.safetensors 2>/dev/null | wc -l) -ge 24 ] && [ -f $wd/stamp.json ] && return 0; stopped && return 1
  momdn $ks $s || return 1
  lock "writers w${kw}m$ks$x"; q2b $wd $kw i8 $wd --only $ONLY --hin $dn/hin --wpre $dn/wpre "${HO[@]}" --seed-offset $s; unlock
  [ $(nq $wd/wq) = 48 ] || { log "QUANT FAIL w${kw}m$ks$x"; return 1; }; rm -rf $wd/wq; }
# dec LEVEL TAG: the 48 writers of a shard dir, decoded as shipped (deq.py, fp16 scales), CPU -> O/dec-TAG
dec() { local d=$O/dec-$2; [ $(nq $d) = 48 ] && return 0
  CUDA_VISIBLE_DEVICES= DEV=cpu OMP_NUM_THREADS=4 nice -n 19 python3 $DEQ wq $1 $d.part $ONLY > $d.log 2>&1 && [ $(nq $d.part) = 48 ] \
    && { rm -rf $d; mv $d.part $d; } || { log "DEQ FAIL $2"; return 1; }; }
# mix TAG SIBWQ DEC: the siblings' 150 weights with the 48 writers replaced -> O/mix-TAG
mix() { local d=$O/mix-$1; [ $(nq $d) = 150 ] && [ -f $d/.ok ] && return 0; rm -rf $d; mkdir -p $d
  for f in $2/*.safetensors; do ln -s $(readlink -f $f) $d/; done
  for f in $3/*.safetensors; do ln -sfn $(readlink -f $f) $d/; done
  local nw=0; for f in $3/*.safetensors; do [ "$(readlink -f $d/$(basename $f))" = "$(readlink -f $f)" ] && nw=$((nw + 1)); done
  [ $(nq $d) = 150 ] && [ $nw = 48 ] || { log "MIX FAIL $1 ($(nq $d) files, $nw writers)"; return 1; }; touch $d/.ok; }
# evs TAG: search + verify of mix-TAG
evs() { stopped && return 1; lock "eval $1"; ev $1 $O/mix-$1 $GS && ev vf-$1 $O/mix-$1 $GV; local rc=$?; unlock; return $rc; }

# ---- gate (seed 0) -------------------------------------------------------------------------------------------------
gate() { [ -f $O/.gate ] && return 0
  sib 3 0 || return 1
  dec $D12/q-k3st k3st && dec $D12/q-k3ste k3ste || return 1
  mix u3 $O/q-k3_swj/wq $O/dec-k3st && mix p3 $O/q-k3_swj/wq $O/dec-k3ste || return 1
  evs u3 && evs p3 || return 1
  pair d1112:k3st u3 $GS $D12/k3st.f32; pair d1112:k3ste_swj p3 $GS $D12/k3ste_swj.f32
  local same=0 n=0; for f in $O/q-k3/q-L*.safetensors; do n=$((n + 1)); cmp -s $f $D12/q-k3ste/$(basename $f) && same=$((same + 1)); done
  log "gate: fresh k3 shards byte-identical to d1112 q-k3ste: $same / $n"
  CUDA_VISIBLE_DEVICES= DEV=cpu OMP_NUM_THREADS=4 nice -n 19 python3 $DEQ check $O/q-k3 $O/q-k3/wq 12 > $O/deqcheck.log 2>&1; log "gate: $(tail -n 1 $O/deqcheck.log)"
  local v; v=$(python3 $R64 gate $O); local rc=$?; log "$v"
  [ $rc = 0 ] && touch $O/.gate || { log "stopping on the gate (w64.md item 1)"; touch $O/STOP; return 1; }; }

# ---- item 1 ----------------------------------------------------------------------------------------------------------
i1seed() { local s=$1 x=$(sfx $1)
  for ks in 2.5 3.5; do sib $ks $s || return 1; done
  dec $D12/q-k3st$x k3st$x && dec $D12/q-k3ste$x k3ste$x && dec $D12/q-k2st$x k2st$x && dec $D12/q-k2ste$x k2ste$x || return 1
  wasym 3 2.5 $s && wasym 3 3.5 $s && wasym 2 2.5 $s || return 1
  dec $O/q-w3m2.5$x w3m2.5$x && dec $O/q-w3m3.5$x w3m3.5$x && dec $O/q-w2m2.5$x w2m2.5$x || return 1
  local c kw ks
  for c in "3 2.5" "3 3.5" "2 2.5"; do kw=${c% *} ks=${c#* }; local sw=$O/q-k$ks${x}_swj/wq
    mix xc$kw-$ks$x $sw $O/dec-k${kw}ste$x && mix xs$kw-$ks$x $sw $O/dec-k${kw}st$x && mix xm$kw-$ks$x $sw $O/dec-w${kw}m$ks$x || return 1
    for a in xc xs xm; do evs $a$kw-$ks$x || return 1; done
    pair xc$kw-$ks$x xs$kw-$ks$x $GS; pair xc$kw-$ks$x xm$kw-$ks$x $GS
    pair vf-xc$kw-$ks$x vf-xs$kw-$ks$x $GV; pair vf-xc$kw-$ks$x vf-xm$kw-$ks$x $GV; done
  log "item 1 seed $s done"; }

# ---- item 3 ----------------------------------------------------------------------------------------------------------
# base K S: the uniform stacks u{K} (asym writers) and p{K} (plain) on fresh K siblings, scored search + verify (+ fit at s0)
base3() { local k=$1 s=$2 x=$(sfx $2); sib $k $s || return 1
  dec $D12/q-k${k}st$x k${k}st$x && dec $D12/q-k${k}ste$x k${k}ste$x || return 1
  mix u$k$x $O/q-k$k${x}_swj/wq $O/dec-k${k}st$x && mix p$k$x $O/q-k$k${x}_swj/wq $O/dec-k${k}ste$x || return 1
  evs u$k$x && evs p$k$x || return 1; momdn $k $s; }
# c9 TAG BASE ALPHA [DN]: c9w.py on mix-BASE -> mix-TAG (CPU, ~10-30 s, no lock)
c9() { [ -f $O/mix-$1/c9w.jsonl ] && return 0; rm -rf $O/mix-$1.part
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python3 $C9W $O/mix-$2 $O/mix-$1 --alpha $3 ${4:+--dn $4} --device cpu >> $O/c9w.log 2>&1 \
    || { log "C9W FAIL $1"; return 1; }; }
i3screen() { local k st fam a
  for k in 3 2; do base3 $k 0 || return 1
    for st in u p; do local dn=; [ $st = u ] && dn=$O/dn-k$k
      stopped && return 1; lock "fit base $st$k"; ev fit-$st$k $O/mix-$st$k $F; unlock
      for fam in attn_o gdn_out down; do for a in -1 1 2 3; do local t=$st$k-$fam$a; stopped && return 1
        [ -f $O/fit-$t.f32 ] || { c9 $t $st$k "$fam=$a" $dn || return 1; lock "screen $t"; ev fit-$t $O/mix-$t $F; local rc=$?; unlock; [ $rc = 0 ] || return 1; }
        pair fit-$st$k fit-$t $F; rm -rf $O/mix-$t; done; done; done; done
  log "item 3 screen done"; }
i3confirm() { local k st s x
  python3 $R64 pick $O > $O/alpha.json.part && mv $O/alpha.json.part $O/alpha.json || { log "PICK FAIL"; return 1; }
  log "alpha*: $(cat $O/alpha.json)"
  for s in 0 1 2; do x=$(sfx $s); for k in 3 2; do base3 $k $s || return 1
    for st in u p; do local al=$(python3 -c "import json; print(json.load(open('$O/alpha.json'))['$st$k'])")
      [ "$al" = "none" ] && { log "item 3: $st$k alpha* all zero: no confirm arm"; continue; }
      local dn=; [ $st = u ] && dn=$O/dn-k$k$x; local t=$st$k${x}c9
      stopped && return 1; c9 $t $st$k$x "$al" $dn || return 1
      evs $t || return 1; pair $st$k$x $t $GS; pair vf-$st$k$x vf-$t $GV; done; done
    log "item 3 seed $s done"; done; }
report() { python3 $R64 report $O > $O/report.md.part 2>> $O/report.err && mv $O/report.md.part $O/report.md; }

if want gate; then gate || exit 1; fi
if want i1; then for s in 0 1 2; do i1seed $s || { log "item 1 seed $s incomplete"; break; }; report; done; fi
if want i3s; then i3screen || log "item 3 screen incomplete"; report; fi
if want i3c; then i3confirm || log "item 3 confirm incomplete"; fi
want report && report
stopped && { log "stopped (STOP)"; exit 0; }
python3 $R64 done $O && log "G64 DONE" || log "G64 INCOMPLETE (see FAIL lines; rerun to retry)"
