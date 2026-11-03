#!/bin/bash
# run.sh [b1|c9|c10|all] -- w02 fixed-code scale experiments on the 0.8B (README.md: ranked.md B1, C9, C10). 4070 box,
# one GPU, allocator capped. Codes never change: every arm rescales an existing quant's dequantized weights
# (/data/yb/q-<src>/wq), so one seed per comparison suffices (B1 runs all three anyway). Idempotent: an arm whose score
# file exists is skipped, a half-written arm dir is redone. Results: /data/yb/w02-fixedscale/result.txt.
X=${X:-/workspace/rad-exl3-mix/proxy2b/exp/w02-fixedscale} P=${P:-/workspace/rad-exl3-mix/proxy2b} M=/data/models/qwen35-08b
Y=${Y:-/data/yb} O=${O:-/data/yb/w02-fixedscale} HO=/data/yb/dv8192/hout
export GS=${GS:-/data/g08-search1} GF=${GF:-/data/g08-fit1}       # (overrides: selftest-run.sh's stubbed dry run)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
die() { log "FAIL $*"; exit 1; }
src() { [ -f $Y/$1.f32 ] && [ $(ls $Y/q-$1/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || die "source $1 missing"; }
# ev OUT.f32 GOLDEN WQDIR NAME
ev() { [ -f $1 ] && return
  log "$4: $(python3 $P/evalq.py $M $2 --wq $3 --out $1 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"
  [ -f $1 ] || die "eval $4"; }
# rf TAG SRC refitq-args...: O/q-TAG from Y/q-SRC (or O/q-SRC)
rf() { local t=$1 sd=$Y/q-$2; shift 2; [ -f $O/q-$t/.done ] && return
  [ -d $sd/wq ] || sd=$O/q-$(basename $sd | cut -c3-)
  rm -rf $O/q-$t; python3 $X/refitq.py $sd $O/q-$t "$@" > $O/q-$t.log 2>&1 || die "refitq $t"; }
pair() { log "pair $(basename $2 .f32) -> $(basename $3 .f32): $(python3 $P/pcmp.py $1 $2 $3 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
want() { [ "$STAGE" = all ] || [ "$STAGE" = $1 ]; }
STAGE=${1:-all}

if want b1; then   # B1: H_out-weighted, undamped, 6-round refit (h6); u6 = the same without h. CPU predictor (README): ~inert,
  # so by default seed 0 only (4 evals); B1_FULL=1 runs the 3-seed rule's 18 evals
  for k in 3 2; do for i in 0 ${B1_FULL:+1 2}; do s=$([ $i = 0 ] || echo _s$i)
    for b in bdiag base; do src k$k$b$s; t=k$k$b${s}_h6
      rf $t k$k$b$s --rounds 6 --undamped --hout $HO; ev $O/$t.f32 $GS $O/q-$t/wq $t; pair $GS $Y/k$k$b$s.f32 $O/$t.f32; done
    [ -n "$B1_FULL" ] || continue
    t=k${k}bdiag${s}_u6; rf $t k${k}bdiag$s --rounds 6 --undamped; ev $O/$t.f32 $GS $O/q-$t/wq $t; pair $GS $Y/k${k}bdiag$s.f32 $O/$t.f32
  done; done
  python3 $X/pstat.py b1 $O $Y | tee -a $O/result.txt
fi

if want c9; then   # C9: one alpha per (level, family) on svh, tuned on the fit slice, confirmed on search + seed 1
  for k in 3 2; do b=k${k}base; src $b; src ${b}_s1
    ev $O/$b.fit.f32 $GF $Y/q-$b/wq $b.fit
    for a in -1 1 2; do t=${b}_aall$a; rf $t $b --alpha all=$a; ev $O/$t.fit.f32 $GF $O/q-$t/wq $t.fit; pair $GF $O/$b.fit.f32 $O/$t.fit.f32; done
    for f in attn_in attn_o gdn_in gdn_out gate_up down; do for a in -1 1 2; do t=${b}_a$f$a
      rf $t $b --alpha $f=$a; ev $O/$t.fit.f32 $GF $O/q-$t/wq $t.fit; done; done
    A=$(python3 $X/pstat.py pick $O $b) || die "pick $b"; log "$b alpha* = $A"
    t=${b}_astar; rf $t $b --alpha $A; ev $O/$t.fit.f32 $GF $O/q-$t/wq $t.fit; pair $GF $O/$b.fit.f32 $O/$t.fit.f32
    ev $O/$t.f32 $GS $O/q-$t/wq $t; pair $GS $Y/$b.f32 $O/$t.f32
    t=${b}_s1_astar; rf $t ${b}_s1 --alpha $A; ev $O/$t.f32 $GS $O/q-$t/wq $t; pair $GS $Y/${b}_s1.f32 $O/$t.f32
  done
  python3 $X/pstat.py c9 $O $Y | tee -a $O/result.txt
fi

if want c10; then  # C10: gate/up output scales fitted jointly through the SwiGLU on fit-pool activations
  for k in 3 2; do for b in base bdiag; do s=k$k$b; src $s
    if [ ! -f $O/q-${s}_swj/.done ] || [ ! -f $O/q-${s}_swu/.done ]; then
      rm -rf $O/q-${s}_swj $O/q-${s}_swu
      python3 $X/swiglufit.py $M $Y/q-$s $GF $O/q-${s}_swj $O/q-${s}_swu > $O/q-${s}_swj.log 2>&1 || die "swiglufit $s"
    fi
    for v in swj $([ $b = base ] && echo swu); do t=${s}_$v; ev $O/$t.f32 $GS $O/q-$t/wq $t; pair $GS $Y/$s.f32 $O/$t.f32; done
    [ $b = base ] && pair $GS $O/${s}_swu.f32 $O/${s}_swj.f32
  done; done
  python3 $X/pstat.py c10 $O $Y | tee -a $O/result.txt
fi
log "W02 $STAGE DONE"
