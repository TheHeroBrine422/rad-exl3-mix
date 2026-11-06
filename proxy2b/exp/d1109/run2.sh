#!/bin/bash
# run2.sh -- d1109 amendments 1 and 2 (README.md, #92), 4070, after run.sh: waits for run.sh's process to exit, then
# runs the arm groups run.sh lacks. Every GPU unit holds /data/yb/gpu.lock for < 15 min. Idempotent: rerun to continue.
# Steps (D1109_STEPS2, default "healrq rqp healrqp summ2"):
#   healrq   amendment 1: d1105's single-pass scale healing (sp: heal.py, qad, 2,048 windows, 1 epoch, 64 per doc, then
#            fold.py) on k{K}rqswjasym_r0, K3 / K2 x seeds 0 / 1 -> k{K}rqswjasym_r0_sp, paired vs the stack's healed
#            d1105 k{K}edyswjasym_r0_sp. heal.py trains in holds of HEAL_HOLD s and resumes from its state.pt
#   rqp      amendment 2: run.sh's rq chain with realq.py --prefix bf16 (every layer quantized on bf16 inputs),
#            K3 / K2 x seeds 0 / 1 -> k{K}rqp, k{K}rqpswjasym_r0; vs the stack, vs rq, bare vs k{K}edy and k{K}rq
#   healrqp  the sp healing on k{K}rqpswjasym_r0 -> k{K}rqpswjasym_r0_sp, vs the stack's and rq's healed arms
#   summ2    summ.py lines and rules.py (the base rule, amendment 1, amendment 2) -> result.txt
# Writers: D1109_WRITERS, default plain (= what run.sh runs with, amendment 3). Stop: touch /data/yb/night.stop.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X=$P/exp/d1109 X2=$P/exp/w02-fixedscale
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
X=$P/exp/d1109   # d1103's sourced block resets X
O=/data/yb/d1109 D3=/data/yb/d1103 D4=/data/yb/d1104 D5=/data/yb/d1105; mkdir -p $O
FISH=$O/fisher RQ_HOLD=${RQ_HOLD:-780} WR=${D1109_WRITERS:-plain} HEAL_HOLD=${HEAL_HOLD:-840} RQX=()
f32() { for d in $O $D4 $D3 $Y $D5; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D4 $D3 $Y; do [ $(nq $d/q-$1/wqall) -ge 150 ] && { echo $d/q-$1/wqall; return; }; [ $(nq $d/q-$1/wq) -ge 150 ] && { echo $d/q-$1/wq; return; }; done; }   # finished dirs only (d1103 nq)
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
want() { [[ " ${D1109_STEPS2:-healrq rqp healrqp summ2} " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
# rqq TAG K [realq opts] -- [ourq args]: REAL-Q quant -> O/q-TAG, in GPU holds of RQ_HOLD s (exit 3 = resume next hold)
rqq() { t=$1 k=$2; shift 2; d=$O/q-$t
  while [ ! -f $d/.rqdone ]; do [ -f /data/yb/night.stop ] && return 1
    lock; FISHER=$FISH/fisher.pt timeout 880 $X/rq.sh $k i8 $d --max-sec $RQ_HOLD "$@" >> $d.log 2>&1; rc=$?; unlock
    case $rc in 0) grep -q "^REALQ DONE" $d.log && touch $d/.rqdone;; 3|124) ;; *) log "REALQ FAIL $t (rc $rc)"; return 1;; esac
    [ $rc = 0 ] && ! [ -f $d/.rqdone ] && { log "REALQ FAIL $t (no DONE)"; return 1; }
  done; log "$t: realq $(python3 -c "import json; r=[json.loads(l) for l in open('$d/realq.jsonl')]; print(len(r), 'linear runs', sum(x['steps'] for x in r), 'steps', round(sum(x['sec'] for x in r) / 60, 1), 'min of quant')")"; }
evl() { [ -f $O/$1.f32 ] && return; lock; ev "$@"; unlock; }
# chain TAG K BASE_STACK SEED [HO args]: run.sh's chain (REAL-Q full quant with RQX's realq options, swj, moments,
# writers asym ρ0) -> TAG + swjasym_r0
chain() { t=$1 k=$2 st=$3 s=$4; shift 4; local ho=("$@")
  rqq $t $k "${RQX[@]}" -- "${ho[@]}" --seed-offset $s || return 1
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
      rqq $a $k "${RQX[@]}" --base-wq $bw -- --only $ONLY --hin $dn/hin $wp "${ho[@]}" --seed-offset $s || return 1
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
# heal1 TAG STACK ARGS..: d1105's heal1 (heal.py train, fold.py, eval with the folded norms) on STACK's weights ->
# O/h-TAG, TAG.f32; training in holds of HEAL_HOLD s (timeout; heal.py resumes from state.pt, saved every 32 steps)
heal1() { local t=$1 st=$2; shift 2; local d=$O/h-$t; [ -f $O/$t.f32 ] && return; local w=$(wqd $st)
  [ -n "$w" ] || { log "no stack $st"; return 1; }; mkdir -p $d
  while [ ! -f $d/.healdone ]; do [ -f /data/yb/night.stop ] && return 1
    lock; timeout $HEAL_HOLD python3 $E6/heal.py train $d --wq $w "$@" >> $d.log 2>&1; rc=$?; unlock
    case $rc in 0) grep -q "^HEAL DONE" $d.log && [ -f $d/params.pt ] && touch $d/.healdone
                   [ -f $d/.healdone ] || { log "heal FAIL $t (no DONE)"; return 1; };;
                124) ;; *) log "heal FAIL $t (rc $rc)"; return 1;; esac
  done
  log "$t: $(grep '^HEAL DONE' $d.log | tail -1)"
  lock
  [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $w $d $M >> $d.log 2>&1 || { log "fold FAIL $t"; unlock; return 1; }
  ev $t $d/wq $S "" $(hm $d); unlock; }
HEAL="--epochs 1 --ntrain 2048 --per-doc 64"   # d1105's sp
# wait for run.sh (same O, same GPU queue); a restart of run2 after run.sh has ended skips this
if pgrep -f "^bash /workspace/rad-exl3-mix/proxy2b/exp/d1109/[r]un.sh( |$)" > /dev/null; then
  log "run2: waiting for run.sh to exit"
  while pgrep -f "^bash /workspace/rad-exl3-mix/proxy2b/exp/d1109/[r]un.sh( |$)" > /dev/null; do sleep 120; done; fi
if want healrq; then step "healrq (amendment 1)"
  for k in 3 2; do for s in 0 1; do B=k${k}rqswjasym_r0; h=${B}_sp$(sfx $s)
    heal1 $h $B$(sfx $s) $HEAL && { pair k${k}edyswjasym_r0_sp$(sfx $s) $h; pair $B$(sfx $s) $h; }
  done; done; fi
if want rqp; then RQX=(--prefix bf16)
  for s in 0 1; do for k in 3 2; do step "rqp K$k s$s (amendment 2)"; t=k${k}rqp$(sfx $s)
    chain $t $k k${k}edyswjasym_r0 $s $HO && { pair k${k}edy$(sfx $s) $t; pair k${k}rq$(sfx $s) $t
      pair k${k}rqswjasym_r0$(sfx $s) k${k}rqpswjasym_r0$(sfx $s); }
  done; done; RQX=(); fi
if want healrqp; then step "healrqp (amendment 2)"
  for k in 3 2; do for s in 0 1; do B=k${k}rqpswjasym_r0; h=${B}_sp$(sfx $s)
    heal1 $h $B$(sfx $s) $HEAL && { pair k${k}edyswjasym_r0_sp$(sfx $s) $h; pair k${k}rqswjasym_r0_sp$(sfx $s) $h; pair $B$(sfx $s) $h; }
  done; done; fi
if want summ2; then step summ2
  for k in 3 2; do python3 $X3/summ.py $O k${k}edyswjasym_r0 k${k}rqswjasym_r0 k${k}edyswjasym_r0_sp k${k}rqswjasym_r0_sp \
    k${k}edyswjasym_r0 k${k}rqpswjasym_r0 k${k}edyswjasym_r0_sp k${k}rqpswjasym_r0_sp \
    k${k}rqswjasym_r0 k${k}rqpswjasym_r0 k${k}rqswjasym_r0_sp k${k}rqpswjasym_r0_sp \
    k${k}rqswjasym_r0 k${k}rqswjasym_r0_sp k${k}rqpswjasym_r0 k${k}rqpswjasym_r0_sp k${k}rq k${k}rqp | tee -a $O/result.txt; done
  python3 $X/rules.py $O | tee -a $O/result.txt; fi
log "D1109 RUN2 DONE"
