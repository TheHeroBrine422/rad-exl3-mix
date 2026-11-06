#!/bin/bash
# run.sh -- d1112 (README.md): xcorp H_in (hi-X0.5) on the full adopted 0.8B stack, K2/K3 (item 1) and K4/K5 (item 2),
# and the fractional-K LDLQ drift (item 3), 4070. Idempotent (rerun = continue: arms are skipped by their score files,
# ourq resumes from its shards); every GPU unit holds /data/yb/gpu.lock (< 15 min). Self-contained helpers (no
# sourced d1103 block: its ev / q / pair use globals that callers clobbered in d1111's run_v).
# Steps (D1112_STEPS, default "prep i1a i2a i3 i1b i2b i1c report"):
#   prep    copy d1111's hi-X0.5 into O (read-only source), check the inputs
#   i1a     item 1, K3 / K2, seeds 0-2: stack (st) and stack + X (sx), four goldens, pairs
#   i2a     item 2, K4 / K5, seeds 0-2
#   i3      item 3, K2.25 / 2.75 / 1.75 / 1.25, seeds 0-2: edy quant, interpolated drift (default) vs OURQ_DRIFT_EXACT=1
#   i1b     item 1 seeds 3-4 (the decision read is seeds 0-4)    i2b  item 2 seeds 3-4    i1c  item 1 seeds 5-6
#   report  report.py -> O/report.md (also after every item)
# Stop: touch /data/yb/d1112/STOP or /data/yb/night.stop (checked between units).
P=/workspace/rad-exl3-mix/proxy2b X2=$P/exp/w02-fixedscale D7=$P/exp/w07-denoise
SNAP=$(dirname "$(readlink -f "$0")")
M=/data/models/qwen35-08b F=/data/g08-fit1 H=/data/h08/calib META=/data/challenge/calib-meta
O=${D1112_OUT:-/data/yb/d1112} HX0=/data/yb/d1111/hi-X0.5
C18=/data/yb/c18/hdiag-snap1024-dy ONLY=/data/yb/w07-seqsib/only.txt
GS=/data/g08-search1 GV=/data/g08-verify1 GX=/data/g08-xcorp2 GA=/data/yb/d1108/g-agent
HO=(--hout $C18 --sigma-reg-out 0.01)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
STEPS=${D1112_STEPS:-prep i1a i2a i3 i1b i2b i1c report}
HX=$O/hi-X0.5
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
stopped() { [ -f $O/STOP ] || [ -f /data/yb/night.stop ]; }
want() { [[ " $STEPS " == *" $1 "* ]] && ! stopped; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
nq() { ls $1/*.safetensors 2>/dev/null | wc -l; }
# ev TAG WQ GOLDEN: O/TAG.f32 (evalq writes it .part -> rename), caller holds the lock
ev() { local tag=$1 wq=$2 g=$3 r; [ -f $O/$tag.f32 ] && return 0
  r=$(python3 $P/evalq.py $M $g --wq $wq --out $O/$tag.f32 2>> $O/eval.err | tail -n 1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')
  [ -f $O/$tag.f32 ] || { log "EVAL FAIL $tag"; return 1; }; log "$tag: $r"; }
# pair A B GOLDEN [A.f32]: "pair A -> B" line (summ.py / report.py format), once
pair() { local a=$1 b=$2 g=$3 fa=${4:-$O/$1.f32}
  grep -qF "pair $a -> $b: \"ratio\"" $O/result.txt 2>/dev/null && return 0
  [ -f $fa ] && [ -f $O/$b.f32 ] || return 1
  log "pair $a -> $b: $(python3 $P/pcmp.py $g $fa $O/$b.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
q2b() { local d=$1; shift; $P/q2b.sh 08 "$@" > $d.log 2>&1; }   # q2b DIR K CB args..: output DIR, log DIR.log
scored() { local t=$1; for x in "" vf- xc- ag-; do [ -f $O/$x$t.f32 ] || return 1; done; }

# chain K SEED ARM: the stack (ARM st: calib H_in; sx: hi-X0.5 for the trunk quant and the writers' G) -> scored
chain() { local k=$1 s=$2 a=$3 hin=() hb=$H; [ $a = sx ] && { hin=(--hin $HX); hb=$HX; }
  local x=$(sfx $s); local b=k$k${a}e$x t=k$k$a$x; local qd=$O/q-$b sw=$O/q-${b}_swj mo=$O/mom-$t dn=$O/dn-$t wd=$O/q-$t
  stopped && return 1
  if ! scored $t; then
    # 1. trunk quant (edy)
    if [ ! -f $sw/.done ] && [ $(nq $qd/wq) -lt 150 ]; then
      lock; q2b $qd $k i8 $qd "${hin[@]}" "${HO[@]}" --seed-offset $s; unlock
      [ $(nq $qd/wq) -ge 150 ] || { log "QUANT FAIL $b"; return 1; }; fi
    [ -f $O/$b.f32 ] || { lock; ev $b $qd/wq $GS; unlock; }
    # 2. swj (its untouched tensors are links into the trunk wq)
    if [ ! -f $sw/.done ]; then mkdir -p $O/q-${b}src; ln -sfn $qd/wq $O/q-${b}src/wq; rm -rf $sw $O/q-${b}_swu
      lock; python3 $X2/swiglufit.py $M $O/q-${b}src $F $sw $O/q-${b}_swu > $sw.log 2>&1; unlock
      [ -f $sw/.done ] || { log "SWIGLUFIT FAIL $b"; return 1; }; fi
    [ -f $O/${b}_swj.f32 ] || { lock; ev ${b}_swj $sw/wq $GS; unlock; }
    stopped && return 1
    # 3. moments through the swj weights (two layer passes, one hold)
    if [ ! -f $mo.done ]; then lock
      for r in 0-11 12-23; do python3 $D7/moments.py $M $META $mo --layers $r --tokens 1000000 --mode sib --wq $sw/wq --seed 0 >> $mo.log 2>&1 \
        || { unlock; log "MOMENTS FAIL $t $r"; return 1; }; done
      unlock; touch $mo.done; log "moments $t: $(grep '^MOMENTS DONE' $mo.log | tail -n 1)"; fi
    # 4. G, F (lambda 1, rho 0) on the arm's H_in
    if [ ! -f $dn.done ]; then lock; python3 $D7/mkdenoise.py $hb $mo $dn --lam 1 --rho 0 --ckpt $M > $dn.log 2>&1; local rc=$?; unlock
      [ $rc = 0 ] || { log "MKDENOISE FAIL $t"; return 1; }; touch $dn.done; fi
    stopped && return 1
    # 5. writers asym rho 0, assembled over the swj weights
    if [ $(nq $wd/wq) -lt 48 ]; then
      lock; q2b $wd $k i8 $wd --only $ONLY --hin $dn/hin $([ -d $dn/wpre ] && echo --wpre $dn/wpre) "${HO[@]}" --seed-offset $s; unlock
      [ $(nq $wd/wq) = 48 ] || { log "QUANT FAIL $t (writers)"; return 1; }; fi
    mkdir -p $wd/wqall
    for f in $sw/wq/*.safetensors; do ln -sfn $(readlink -f $f) $wd/wqall/; done
    for f in $wd/wq/*.safetensors; do ln -sfn $f $wd/wqall/; done
    [ $(nq $wd/wqall) = 150 ] || { log "ASSEMBLE FAIL $t"; return 1; }
    # 6. the four goldens
    for p in :$GS vf-:$GV xc-:$GX ag-:$GA; do stopped && return 1; lock; ev ${p%%:*}$t $wd/wqall ${p#*:}; unlock; done
    scored $t || return 1
  fi
  # 7. scored: drop the arm's big intermediates (wq copies, moments, G / F); shards, logs and scores stay
  [ -f $wd/.clean ] || { rm -rf $qd/wq $sw/wq $O/q-${b}_swu $O/q-${b}src $mo $dn/hin $dn/wpre $wd/wq $wd/wqall; touch $wd/.clean; }
}
# stackpair K SEED: sx vs st on every golden, plus the stage readouts; reproduction check vs d1104 (K2 / K3)
stackpair() { local k=$1 x=$(sfx $2)
  pair k${k}st$x k${k}sx$x $GS; pair vf-k${k}st$x vf-k${k}sx$x $GV; pair xc-k${k}st$x xc-k${k}sx$x $GX; pair ag-k${k}st$x ag-k${k}sx$x $GA
  pair k${k}ste$x k${k}sxe$x $GS; pair k${k}ste${x}_swj k${k}sxe${x}_swj $GS; pair k${k}ste$x k${k}st$x $GS
  local r=/data/yb/d1104/k${k}edyswjasym_r0$x.f32; [ -f $r ] && pair d1104:k${k}edyswjasym_r0$x k${k}st$x $GS $r; }
item() { local ks=$1 ss=$2 k s; for s in $ss; do for k in $ks; do stopped && return 1
  chain $k $s st; chain $k $s sx; stackpair $k $s; done; done; report; }
# fq K SEED MODE: item 3 edy trunk quant, MODE i (default: interpolated drift) | x (OURQ_DRIFT_EXACT=1), search only
fq() { local k=$1 s=$2 m=$3; local t=f$k$m$(sfx $s); local d=$O/q-$t
  [ -f $O/$t.f32 ] || { stopped && return 1
    if [ $(nq $d/wq) -lt 150 ]; then lock
      ( [ $m = x ] && export OURQ_DRIFT_EXACT=1; q2b $d $k i8 $d "${HO[@]}" --seed-offset $s )
      unlock; [ $(nq $d/wq) -ge 150 ] || { log "QUANT FAIL $t"; return 1; }; fi
    log "$t drift: $(head -n 1 $d/proxy.jsonl | grep -o '"drift": [0-9.]*')"
    lock; ev $t $d/wq $GS; unlock; }
  [ -f $O/$t.f32 ] && rm -rf $d/wq; }
report() { python3 $SNAP/report.py $O > $O/report.md.part 2>> $O/report.err && mv $O/report.md.part $O/report.md; }

if want prep; then
  [ -f $HX/.done ] || { rm -rf $HX.part; cp -r $HX0 $HX.part && [ -f $HX.part/.done ] && mv $HX.part $HX || { log "PREP FAIL: copy $HX0"; exit 1; }; }
  for p in $M $F $H $META $C18 $ONLY $GS $GV $GX $GA /data/yb/gpu.lock; do [ -e $p ] || { log "PREP FAIL: missing $p"; exit 1; }; done
  [ -f $O/.prep ] || { log "prep ok: hi-X0.5 $(ls $HX/*.safetensors | wc -l) files, code $(git -C /workspace/rad-exl3-mix rev-parse --short HEAD 2>/dev/null) $(git -C /workspace/rad-exl3-mix status --porcelain quantize proxy2b 2>/dev/null | wc -l) dirty"; touch $O/.prep; }
fi
want i1a && { log "== item 1, seeds 0-2"; item "3 2" "0 1 2"; }
want i2a && { log "== item 2, seeds 0-2"; item "4 5" "0 1 2"; }
if want i3; then log "== item 3"
  for s in 0 1 2; do for k in 2.25 2.75 1.75 1.25; do fq $k $s i; fq $k $s x; pair f${k}x$(sfx $s) f${k}i$(sfx $s) $GS; done; done; report; fi
want i1b && { log "== item 1, seeds 3-4"; item "3 2" "3 4"; }
want i2b && { log "== item 2, seeds 3-4"; item "4 5" "3 4"; }
want i1c && { log "== item 1, seeds 5-6"; item "3 2" "5 6"; }
want report && report
stopped && { log "stopped (STOP / night.stop)"; exit 0; }
# DONE only when every arm of every item is scored (a failed unit leaves the queue not-done: the watchdog sees it)
miss=0
for s in 0 1 2 3 4 5 6; do for k in 3 2; do for a in st sx; do scored k$k$a$(sfx $s) || miss=$((miss + 1)); done; done; done
for s in 0 1 2 3 4; do for k in 4 5; do for a in st sx; do scored k$k$a$(sfx $s) || miss=$((miss + 1)); done; done; done
for s in 0 1 2; do for k in 2.25 2.75 1.75 1.25; do for m in i x; do [ -f $O/f$k$m$(sfx $s).f32 ] || miss=$((miss + 1)); done; done; done
[ $miss = 0 ] && log "D1112 DONE" || log "D1112 INCOMPLETE: $miss arms missing (see FAIL lines; rerun to retry)"
