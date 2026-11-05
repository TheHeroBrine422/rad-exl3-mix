#!/bin/bash
# run.sh -- d1109 (README.md, #92): REAL-Q on the 0.8B on top of our stack, 4070. Every GPU unit holds /data/yb/gpu.lock
# for < 15 min (REAL-Q quants run in holds of RQ_HOLD s and resume from their per-layer shards). Idempotent: rerun to
# continue. Steps (D1109_STEPS, default "fisher rq rqnoh summ"):
#   fisher  stage 0: aggregated block-output Fisher, bf16 model, 1,024 qad windows (as c18), gen-masked (fisher.py)
#   rq      K3 / K2 x seeds 0-2: REAL-Q with the stack's H_out (c18 Σdy², σ_out 0.01) -> swj (fit) -> moments through
#           it -> writers asym ρ0 requantized by REAL-Q on top (--only, --base-wq; D1109_WRITERS=plain: by plain ourq,
#           as the stack does) -> vs k{K}edyswjasym_r0 (the rule); the bare REAL-Q quant also vs k{K}edy
#   rqnoh   the same without H_out anywhere (writers too) -> vs the stack (reported, no rule); bare vs k{K}base
#   summ    summ.py lines for the rule
# Stop: touch /data/yb/night.stop (checked between units).
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X=$P/exp/d1109 X2=$P/exp/w02-fixedscale
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
X=$P/exp/d1109   # d1103's sourced block resets X
O=/data/yb/d1109 D3=/data/yb/d1103 D4=/data/yb/d1104; mkdir -p $O
FISH=$O/fisher RQ_HOLD=${RQ_HOLD:-780} WR=${D1109_WRITERS:-rq}
f32() { for d in $O $D4 $D3 $Y; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D4 $D3 $Y; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
want() { [[ " ${D1109_STEPS:-fisher rq rqnoh summ} " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
# rqq TAG K [realq opts] -- [ourq args]: REAL-Q quant -> O/q-TAG, in GPU holds of RQ_HOLD s (exit 3 = resume next hold)
rqq() { t=$1 k=$2; shift 2; d=$O/q-$t
  while [ ! -f $d/.rqdone ]; do [ -f /data/yb/night.stop ] && return 1
    lock; FISHER=$FISH/fisher.pt timeout 880 $X/rq.sh $k i8 $d --max-sec $RQ_HOLD "$@" >> $d.log 2>&1; rc=$?; unlock
    case $rc in 0) grep -q "^REALQ DONE" $d.log && touch $d/.rqdone;; 3|124) ;; *) log "REALQ FAIL $t (rc $rc)"; return 1;; esac
    [ $rc = 0 ] && ! [ -f $d/.rqdone ] && { log "REALQ FAIL $t (no DONE)"; return 1; }
  done; log "$t: realq $(python3 -c "import json; r=[json.loads(l) for l in open('$d/realq.jsonl')]; print(len(r), 'linear runs', sum(x['steps'] for x in r), 'steps', round(sum(x['sec'] for x in r) / 60, 1), 'min of quant')")"; }
evl() { [ -f $O/$1.f32 ] && return; lock; ev "$@"; unlock; }
# chain TAG K BASE_STACK [HO args]: REAL-Q full quant TAG, then swj, moments, writers asym ρ0 -> TAG + swjasym_r0
chain() { t=$1 k=$2 st=$3 s=$4; shift 4; local ho=("$@")
  rqq $t $k -- "${ho[@]}" --seed-offset $s || return 1
  [ $(ls $O/q-$t/wq/*.safetensors | wc -l) = 150 ] || { log "INCOMPLETE $t"; return 1; }
  evl $t $O/q-$t/wq
  sw=$O/q-${t}_swj
  if [ ! -f $sw/.done ]; then src=$O/q-${t}src; mkdir -p $src; ln -sfn $O/q-$t/wq $src/wq; rm -rf $sw $O/q-${t}_swu
    lock; python3 $X2/swiglufit.py $M $src $F $sw $O/q-${t}_swu > $sw.log 2>&1; rc=$?; unlock
    [ $rc = 0 ] && [ -f $sw/.done ] || { log "swiglufit FAIL $t"; return 1; }; fi
  bw=$sw/wq; b=${t}_swj; a=${t%$(sfx $s)}swjasym_r0$(sfx $s)
  [ -f $O/mom-$b.done ] || { lock; mom $O/mom-$b --mode sib --wq $bw --seed 0; rc=$?; unlock; [ $rc = 0 ] || return 1; }
  mk $O/mom-$b $O/dn-r0-$b 1 0 || return 1
  dn=$O/dn-r0-$b; wp=$([ -d $dn/wpre ] && echo --wpre $dn/wpre)
  if [ ! -f $(f32 $a) ]; then
    if [ $WR = rq ]; then
      rqq $a $k --base-wq $bw -- --only $ONLY --hin $dn/hin $wp "${ho[@]}" --seed-offset $s || return 1
      [ $(ls $O/q-$a/wq/*.safetensors 2>/dev/null | wc -l) = 48 ] || { log "QUANT FAIL $a"; return 1; }
      w=$O/q-$a/wqall; mkdir -p $w
      for f in $bw/*.safetensors; do ln -sfn $(readlink -f $f) $w/; done
      for f in $O/q-$a/wq/*.safetensors; do ln -sfn $f $w/; done
      [ $(ls $w/*.safetensors | wc -l) = 150 ] || { log "ASSEMBLE FAIL $a"; return 1; }
    else
      lock; qo $a $k $bw --hin $dn/hin $wp "${ho[@]}" --seed-offset $s; rc=$?; unlock; [ $rc = 0 ] || return 1
    fi
    evl $a $O/q-$a/wqall; fi
  pair $st$(sfx $s) $a; }
if want fisher; then step fisher
  while [ ! -f $FISH/fisher.pt ]; do [ -f /data/yb/night.stop ] && exit 0
    lock; timeout 880 python3 $X/fisher.py $M $FISH 1024 --max-sec 780 >> $O/fisher.log 2>&1; rc=$?; unlock
    [ $rc = 0 ] || [ $rc = 3 ] || [ $rc = 124 ] || { log "FISHER FAIL rc $rc"; exit 1; }; done
  log "fisher: $(grep -h 'FISHER DONE' $O/fisher.log | tail -1)"; fi
for s in 0 1 2; do for k in 3 2; do
  if want rq; then step "rq K$k s$s"; t=k${k}rq$(sfx $s)
    chain $t $k k${k}edyswjasym_r0 $s $HO && pair k${k}edy$(sfx $s) $t; fi
  if want rqnoh; then step "rqnoh K$k s$s"; t=k${k}rqnoh$(sfx $s)
    chain $t $k k${k}edyswjasym_r0 $s && pair k${k}base$(sfx $s) $t; fi
done; done
if want summ; then step summ
  for k in 3 2; do python3 $X3/summ.py $O k${k}edyswjasym_r0 k${k}rqswjasym_r0 k${k}edy k${k}rq \
    k${k}edyswjasym_r0 k${k}rqnohswjasym_r0 k${k}base k${k}rqnoh | tee -a $O/result.txt; done
  echo "RULE (README): rq earns a 27B test if k3rqswjasym_r0 and k2rqswjasym_r0 vs k{K}edyswjasym_r0 both have mean ratio <= 0.95 and every seed < 1" | tee -a $O/result.txt; fi
log "D1109 DONE"
