#!/bin/bash
# run.sh -- d1113 (README.md): swj-style joint scale refits at the attention / GDN junctions (jfit.py), 0.8B, 4070.
# Idempotent (rerun = continue: arms are skipped by their score files, ourq resumes from its shards, jfit variants
# by their .done); every GPU unit holds /data/yb/gpu.lock (< 15 min). Self-contained helpers (d1112's pattern).
# Steps (D1113_STEPS, default "prep s1 s1k4 pick f2 f2k4 s1b f2b report"):
#   prep  check the inputs           s1    screen K3 / K2 seeds 0-2 (trunk edy -> swj -> jfit; fit golden; mA on search)
#   s1k4  screen K4 seeds 0-2        pick  report.py --pick -> O/pick.txt (once; never rewritten)
#   f2    stage 2 (full stack: st, fa, fg, fc) K3 / K2 seeds 0-2, search + verify     f2k4  the same at K4
#   s1b / f2b  screen and stage 2 at K3 / K2 seeds 3-4 (robustness)                  report  report.py -> O/report.md
# Stop: touch /data/yb/d1113/STOP or /data/yb/night.stop (checked between units).
P=/workspace/rad-exl3-mix/proxy2b X2=$P/exp/w02-fixedscale D7=$P/exp/w07-denoise
SNAP=$(dirname "$(readlink -f "$0")")
M=/data/models/qwen35-08b F=/data/g08-fit1 H=/data/h08/calib META=/data/challenge/calib-meta
O=${D1113_OUT:-/data/yb/d1113} R12=/data/yb/d1112
C18=/data/yb/c18/hdiag-snap1024-dy ONLY=/data/yb/w07-seqsib/only.txt
GS=/data/g08-search1 GV=/data/g08-verify1
HO=(--hout $C18 --sigma-reg-out 0.01)
VARS="qk vg at gz gd mA aw dw"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
STEPS=${D1113_STEPS:-prep s1 s1k4 pick f2 f2k4 s1b f2b report}
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
# pair A B GOLDEN [A.f32]: "pair A -> B" line (report.py format), once
pair() { local a=$1 b=$2 g=$3 fa=${4:-$O/$1.f32}
  grep -qF "pair $a -> $b: \"ratio\"" $O/result.txt 2>/dev/null && return 0
  [ -f $fa ] && [ -f $O/$b.f32 ] || return 1
  log "pair $a -> $b: $(python3 $P/pcmp.py $g $fa $O/$b.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
q2b() { local d=$1; shift; $P/q2b.sh 08 "$@" > $d.log 2>&1; }   # q2b DIR K CB args..: output DIR, log DIR.log

# base K SEED: trunk edy + swj (d1112's st recipe, steps 1-2) -> O/q-k{K}e{x}_swj/wq; also the swj search score
# (reproduction check vs d1112's k{K}ste{x}_swj). Returns 1 if the base can't be (re)built.
base() { local k=$1 s=$2 x=$(sfx $2); local b=k${k}e$x; local qd=$O/q-$b sw=$O/q-${b}_swj
  stopped && return 1
  if [ ! -f $sw/.done ] || [ $(nq $qd/wq) -lt 150 ]; then
    if [ $(nq $qd/wq) -lt 150 ]; then
      lock; q2b $qd $k i8 $qd "${HO[@]}" --seed-offset $s; unlock
      [ $(nq $qd/wq) -ge 150 ] || { log "QUANT FAIL $b"; return 1; }; fi
    if [ ! -f $sw/.done ]; then mkdir -p $O/q-${b}src; ln -sfn $qd/wq $O/q-${b}src/wq; rm -rf $sw $O/q-${b}_swu
      lock; python3 $X2/swiglufit.py $M $O/q-${b}src $F $sw $O/q-${b}_swu > $sw.log 2>&1; unlock
      [ -f $sw/.done ] || { log "SWIGLUFIT FAIL $b"; return 1; }; rm -rf $O/q-${b}_swu; fi
  fi
  [ -f $O/${b}_swj.f32 ] || { lock; ev ${b}_swj $sw/wq $GS; unlock; }
  local r=$R12/k${k}ste${x}_swj.f32
  [ -f $O/${b}_swj.f32 ] && [ -f $r ] && ! grep -qF "repro ${b}_swj" $O/result.txt && \
    { cmp -s $r $O/${b}_swj.f32 && log "repro ${b}_swj: identical to d1112 k${k}ste${x}_swj" || log "repro ${b}_swj: DIFFERS from d1112 k${k}ste${x}_swj"; }
  return 0; }

# screen K SEED: base -> jfit (all variants) -> fit-golden scores of base and qk vg at gz gd; mA on search (K3 / K2)
screen() { local k=$1 s=$2 x=$(sfx $2); local b=k${k}e$x; local qd=$O/q-$b sw=$O/q-${b}_swj j=$O/j-$b v
  local all=1; for v in qk vg at gz gd aw dw; do [ -f $O/fit-k${k}j$v$x.f32 ] || all=0; done
  [ $k = 4 ] || [ -f $O/k${k}jmA$x.f32 ] || all=0
  if [ $all = 0 ]; then
    base $k $s || return 1; stopped && return 1
    local need=0; for v in $VARS; do [ -f $j/$v/.done ] || need=1; done
    if [ $need = 1 ]; then lock; python3 $SNAP/jfit.py $M $META $sw $qd $j > $j.log 2>&1; local rc=$?; unlock
      [ $rc = 0 ] || { log "JFIT FAIL $b (rc $rc)"; return 1; }
      log "jfit $b: $(grep '^SUMMARY' $j.log | tr '\n' ' ')"; fi
    lock; ev fit-${b}_swj $sw/wq $F   # one hold for the screen's evals (8-9 x ~40 s < 15 min)
    for v in qk vg at gz gd aw dw; do stopped && { unlock; return 1; }; ev fit-k${k}j$v$x $j/$v/wq $F; done
    [ $k = 4 ] || ev k${k}jmA$x $j/mA/wq $GS; unlock
  fi
  for v in qk vg at gz gd aw dw; do pair fit-${b}_swj fit-k${k}j$v$x $F; done
  [ $k = 4 ] || pair ${b}_swj k${k}jmA$x $GS
  return 0; }

# up K SEED ARM: the upstream wq dir an arm's moments / writers see (st: swj; fa / fg: the pick's jfit variant; fc: both)
pick() { grep -o "^$1=[a-z]*" $O/pick.txt 2>/dev/null | cut -d= -f2; }
up() { local k=$1 x=$(sfx $2) a=$3; local b=k${k}e$x; local sw=$O/q-${b}_swj j=$O/j-$b
  case $a in
    st) echo $sw/wq;;
    fa) echo $j/$(pick attn)/wq;;
    fg) echo $j/$(pick gdn)/wq;;
    fc) local d=$O/q-k${k}fc${x}src/wq ja=$j/$(pick attn)/wq jg=$j/$(pick gdn)/wq f n
      if [ $(nq $d) -lt 150 ]; then rm -rf $d; mkdir -p $d
        for f in $sw/wq/*.safetensors; do n=$(basename $f)
          if [ -f $ja/$n ] && [ ! -L $ja/$n ]; then ln -s $ja/$n $d/$n
          elif [ -f $jg/$n ] && [ ! -L $jg/$n ]; then ln -s $jg/$n $d/$n
          else ln -s $(readlink -f $f) $d/$n; fi; done; fi
      echo $d;;
  esac; }

# chain K SEED ARM: moments through the arm's upstream -> G, F -> writers asym rho0 -> wqall -> search + verify
chain() { local k=$1 s=$2 a=$3; local x=$(sfx $s); local t=k$k$a$x b=k${k}e$x
  local mo=$O/mom-$t dn=$O/dn-$t wd=$O/q-$t U
  [ -f $O/$t.f32 ] && [ -f $O/vf-$t.f32 ] && return 0
  stopped && return 1
  if [ $a = st ]; then   # control reuse: same trunk + swj bytes as d1112 -> its st scores are this arm's
    if grep -qF "repro ${b}_swj: identical" $O/result.txt && [ -f $R12/$t.f32 ] && [ -f $R12/vf-$t.f32 ]; then
      cp $R12/$t.f32 $O/$t.f32.part && mv $O/$t.f32.part $O/$t.f32; cp $R12/vf-$t.f32 $O/vf-$t.f32.part && mv $O/vf-$t.f32.part $O/vf-$t.f32
      log "control $t: d1112 score files reused (trunk + swj identical)"; return 0; fi; fi
  base $k $s || return 1
  if [ $a != st ]; then [ -f $O/j-$b/mA/.done ] || screen $k $s || return 1; fi
  U=$(up $k $s $a); [ $(nq $U) = 150 ] || { log "UPSTREAM FAIL $t ($U)"; return 1; }
  if [ $(nq $wd/wq) -lt 48 ]; then
    if [ ! -f $dn.done ]; then
      if [ ! -f $mo.done ]; then lock
        for r in 0-11 12-23; do python3 $D7/moments.py $M $META $mo --layers $r --tokens 1000000 --mode sib --wq $U --seed 0 >> $mo.log 2>&1 \
          || { unlock; log "MOMENTS FAIL $t $r"; return 1; }; done
        unlock; touch $mo.done; log "moments $t: $(grep '^MOMENTS DONE' $mo.log | tail -n 1)"; fi
      lock; python3 $D7/mkdenoise.py $H $mo $dn --lam 1 --rho 0 --ckpt $M > $dn.log 2>&1; local rc=$?; unlock
      [ $rc = 0 ] || { log "MKDENOISE FAIL $t"; return 1; }; touch $dn.done; rm -rf $mo; fi
    stopped && return 1
    lock; q2b $wd $k i8 $wd --only $ONLY --hin $dn/hin $([ -d $dn/wpre ] && echo --wpre $dn/wpre) "${HO[@]}" --seed-offset $s; unlock
    [ $(nq $wd/wq) = 48 ] || { log "QUANT FAIL $t (writers)"; return 1; }
    rm -rf $dn/hin $dn/wpre; fi
  mkdir -p $wd/wqall
  for f in $U/*.safetensors; do ln -sfn $(readlink -f $f) $wd/wqall/; done
  for f in $wd/wq/*.safetensors; do ln -sfn $f $wd/wqall/; done
  [ $(nq $wd/wqall) = 150 ] || { log "ASSEMBLE FAIL $t"; return 1; }
  for p in :$GS vf-:$GV; do stopped && return 1; lock; ev ${p%%:*}$t $wd/wqall ${p#*:}; unlock; done
  [ -f $O/$t.f32 ] && [ -f $O/vf-$t.f32 ] || return 1
  rm -rf $wd/wq $wd/wqall $O/mom-$t; }

arms() { local as=st; [ -n "$(pick attn)" ] && as="$as fa"; [ -n "$(pick gdn)" ] && as="$as fg"
  [ -n "$(pick attn)" ] && [ -n "$(pick gdn)" ] && as="$as fc"; echo $as; }
# clean K SEED: once every planned arm of (K, seed) is scored: drop the trunk / swj / variant wq copies (scales kept)
clean() { local k=$1 x=$(sfx $2) a; local b=k${k}e$x
  for a in $(arms); do [ -f $O/k$k$a$x.f32 ] && [ -f $O/vf-k$k$a$x.f32 ] || return 0; done
  [ -f $O/.clean-$b ] && return 0
  rm -rf $O/q-$b/wq $O/q-${b}_swj/wq $O/q-${b}src $O/q-k${k}fc${x}src; for v in $VARS; do rm -rf $O/j-$b/$v/wq; done
  touch $O/.clean-$b; }
stage2() { local ks=$1 ss=$2 k s a; [ -f $O/pick.txt ] || { log "stage 2: no pick.txt"; return 1; }
  for s in $ss; do for k in $ks; do for a in $(arms); do stopped && return 1; chain $k $s $a; done
    for a in $(arms); do [ $a = st ] && continue; pair k${k}st$(sfx $s) k$k$a$(sfx $s) $GS; pair vf-k${k}st$(sfx $s) vf-k$k$a$(sfx $s) $GV; done
    clean $k $s; done; done; report; }
screens() { local ks=$1 ss=$2 k s; for s in $ss; do for k in $ks; do stopped && return 1; screen $k $s; done; done; report; }
report() { python3 $SNAP/report.py $O > $O/report.md.part 2>> $O/report.err && mv $O/report.md.part $O/report.md; }

if want prep; then
  for p in $M $F $H $META $C18 $ONLY $GS $GV /data/yb/gpu.lock $SNAP/jfit.py $SNAP/report.py; do [ -e $p ] || { log "PREP FAIL: missing $p"; exit 1; }; done
  [ -f $O/.prep ] || { log "prep ok: code $(git -C /workspace/rad-exl3-mix rev-parse --short HEAD 2>/dev/null) $(git -C /workspace/rad-exl3-mix status --porcelain quantize proxy2b 2>/dev/null | wc -l) dirty"; touch $O/.prep; }
fi
want s1 && { log "== screen K3 / K2, seeds 0-2"; screens "3 2" "0 1 2"; }
want s1k4 && { log "== screen K4, seeds 0-2"; screens "4" "0 1 2"; }
if want pick && [ ! -f $O/pick.txt ]; then
  python3 $SNAP/report.py $O --pick > $O/pick.txt.part 2>> $O/report.err && grep -q "^final=1" $O/pick.txt.part \
    && mv $O/pick.txt.part $O/pick.txt && log "pick: $(tr '\n' ' ' < $O/pick.txt)" || log "PICK FAIL (screen incomplete?)"
fi
want f2 && { log "== stage 2 K3 / K2, seeds 0-2: arms $(arms)"; stage2 "3 2" "0 1 2"; }
want f2k4 && { log "== stage 2 K4, seeds 0-2"; stage2 "4" "0 1 2"; }
want s1b && { log "== screen K3 / K2, seeds 3-4"; screens "3 2" "3 4"; }
want f2b && { log "== stage 2 K3 / K2, seeds 3-4"; stage2 "3 2" "3 4"; }
want report && report
stopped && { log "stopped (STOP / night.stop)"; exit 0; }
miss=0
for s in 0 1 2 3 4; do for k in 3 2; do for v in qk vg at gz gd aw dw; do [ -f $O/fit-k${k}j$v$(sfx $s).f32 ] || miss=$((miss + 1)); done; done; done
for s in 0 1 2; do for v in qk vg at gz gd aw dw; do [ -f $O/fit-k4j$v$(sfx $s).f32 ] || miss=$((miss + 1)); done; done
if [ -f $O/pick.txt ]; then
  for s in 0 1 2 3 4; do for k in 3 2; do for a in $(arms); do [ -f $O/vf-k$k$a$(sfx $s).f32 ] || miss=$((miss + 1)); done; done; done
  for s in 0 1 2; do for a in $(arms); do [ -f $O/vf-k4$a$(sfx $s).f32 ] || miss=$((miss + 1)); done; done
else miss=$((miss + 1)); fi
[ $miss = 0 ] && log "D1113 DONE" || log "D1113 INCOMPLETE: $miss arms missing (see FAIL lines; rerun to retry)"
