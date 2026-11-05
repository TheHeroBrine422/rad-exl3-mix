#!/bin/bash
# run.sh -- d1108 (README.md), 0.8B, 4070; GPU units hold /data/yb/gpu.lock. Idempotent. Stop: touch /data/yb/night.stop.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X=$P/exp/d1108 E6=$P/exp/w06-heal
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
O=/data/yb/d1108 D3=/data/yb/d1103 D4=/data/yb/d1104 D5=/data/yb/d1105; mkdir -p $O
f32() { for d in $O $D5 $D4 $D3 $Y; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D5 $D4 $D3 $Y; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
want() { [[ " ${D1108_STEPS:-prep gen build gold coll hout fam hin heal} " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
T=/data/pisess/v2/pisess-redacted.jsonl TD=/data/refs/td-qwen38/cal_trace.json GA=$O/g-agent C23=$Y/c23/hdiag-snap1024-dy
evboth() { t=$1 w=$2 em=${3:-$M}; ev $t $w $S "" $em; ev ag-$t $w $GA "" $em; }   # calib search + agentic
pairboth() { pair $1 $2; pair ag-$1 ag-$2 $GA; }
if want prep; then step prep
  [ -f $O/prep/prep.json ] || CUDA_VISIBLE_DEVICES= python3 -I $X/sgprep.py $T $O/prep > $O/prep.log 2>&1 || { log "prep FAIL"; exit 1; }
  log "prep: $(cat $O/prep/prep.json)"; fi
if want gen; then step gen
  for s in train eval; do [ -f $O/gen-$s/.done ] && continue
    lock; python3 $X/sggen.py $M $O/prep/prompts-$s.npy $O/gen-$s >> $O/gen-$s.log 2>&1 && touch $O/gen-$s/.done; unlock
    log "gen $s: $(tail -2 $O/gen-$s.log | head -1)"; done; fi
if want build; then step build
  for k in selfgen-train:C selfgen-eval:E raw-train:B td-trace:T; do kind=${k%:*} d=$O/m${k#*:}; g=$O/gen-${kind#selfgen-}; [ $kind = td-trace ] && g=$TD
    [ -f $d/build.json ] || CUDA_VISIBLE_DEVICES= python3 -I $X/sgbuild.py $O/prep $g $d $kind >> $O/build.log 2>&1
    log "build $kind: $(cat $d/build.json)"; done; fi
if want gold; then step gold
  if [ ! -f $GA/rows.f32 ]; then n=$(python3 -c "import json; print(len(json.load(open('$O/mE/kld.json'))['docs']))")
    lock; python3 $P/mkgold.py $M $O/mE $GA $(seq 0 $((n - 1))) > $O/gold.log 2>&1; unlock; fi
  log "g-agent: $(python3 -c "import json,numpy as np; m=json.load(open('$GA/kld.json')); print(len(m['docs']), 'docs', int(np.fromfile('$GA/gen.u8',np.uint8).sum()), 'scored')")"; fi
if want coll; then step coll   # Σdy² on C and B, as c23 (gen-masked, 1,024 windows of 2,048), and input Hessians on C
  for c in C B T; do CD=$O/c$c; [ -d $CD/hdiag-snap1024-dy ] && continue
    W=/data/d1108work-$c; mkdir -p $W/challenge/golden; ln -sfn $O/m$c $W/challenge/golden/calib
    lock; RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=$O/m$c/pools.json YAQA_FP32=1 YAQA_DIAG=1 YAQA_DIAG_EXTRA=1 YAQA_GENMASK=1 YAQA_SEED=0 \
      YAQA_SNAP=1024 YAQA_SAVE_EVERY=64 python3 /workspace/rad-exl3-mix/v3/yaqa/collect.py $CD 1024 2048 > $O/c$c.log 2>&1; unlock
    grep -q "^DONE" $O/c$c.log && log "collected $c: $(grep -h '^DONE' $O/c$c.log)" || log "COLLECT $c FAIL"; done
  [ -f $O/hinC/.done ] || { lock; python3 $P/hcollect.py $M $O/mC $O/hinC > $O/hinC.log 2>&1 && touch $O/hinC/.done; unlock; log "hin C: $(tail -1 $O/hinC.log)"; }
fi
if want hout; then step hout
  for b in C:0.25 C:0.5 C:1 B:0.5 B:1 T:0.5 T:1; do c=${b%:*} w=${b#*:}; d=$O/ho-$c$w
    [ -f $d/.done ] || python3 -I $X/blend.py hout $C23 $O/c$c/hdiag-snap1024-dy $w $d >> $O/blend.log 2>&1; done
  for s in 0 1; do for k in 3 2; do base=k${k}dygm$(sfx $s); bw=$(wqd $base)
    lock; [ -n "$bw" ] && evboth $base $bw; unlock
    for b in C0.25 C0.5 C1 B0.5 B1 T0.5 T1; do t=k${k}ho$b$(sfx $s)
      lock; q $t $k i8 --hout $O/ho-$b --sigma-reg-out 0.01 --seed-offset $s; ev ag-$t $O/q-$t/wq $GA; unlock
      pairboth $base $t; done; done; done
  for b in C0.25 C0.5 C1 B0.5 B1 T0.5 T1; do for k in 3 2; do
    python3 $X3/summ.py $O k${k}dygm k${k}ho$b ag-k${k}dygm ag-k${k}ho$b | tee -a $O/result.txt; done; done; fi
if want fam; then step fam   # amendment 2: per-family blends at w* (mixers = attention + GDN, mlp), vs the uniform blend
  w=$(python3 -I $X/wstar.py $O hoC 0.25 0.5 1); log "w* (C, uniform): $w"
  if [ "$w" != none ]; then
    for f in "mix:(self_attn|linear_attn)" "mlp:mlp\\."; do n=${f%%:*} re=${f#*:}; d=$O/ho-C$w-$n
      [ -f $d/.done ] || python3 -I $X/blend.py hout $C23 $O/cC/hdiag-snap1024-dy $w $d "$re" >> $O/blend.log 2>&1
      for s in 0 1; do for k in 3 2; do t=k${k}hoC$w$n$(sfx $s)
        lock; q $t $k i8 --hout $d --sigma-reg-out 0.01 --seed-offset $s; ev ag-$t $O/q-$t/wq $GA; unlock
        pairboth k${k}hoC$w$(sfx $s) $t; pairboth k${k}dygm$(sfx $s) $t; done; done
      for k in 3 2; do python3 $X3/summ.py $O k${k}hoC$w k${k}hoC$w$n ag-k${k}hoC$w ag-k${k}hoC$w$n | tee -a $O/result.txt; done; done
  fi; fi
if want hin; then step hin
  for w in 0.5 1; do d=$O/hi-C$w; [ -f $d/.done ] || python3 -I $X/blend.py hin /data/h08/calib $O/hinC $w $d >> $O/blend.log 2>&1; done
  for s in 0 1; do for k in 3 2; do base=k${k}dygm$(sfx $s)
    for w in 0.5 1; do t=k${k}hiC$w$(sfx $s)
      lock; q $t $k i8 --hin $O/hi-C$w --hout $C23 --sigma-reg-out 0.01 --seed-offset $s; ev ag-$t $O/q-$t/wq $GA; unlock
      pairboth $base $t; done; done; done
  for w in 0.5 1; do for k in 3 2; do python3 $X3/summ.py $O k${k}dygm k${k}hiC$w ag-k${k}dygm ag-k${k}hiC$w | tee -a $O/result.txt; done; done; fi
if want heal; then step heal
  for k in 3 2; do for s in 0 1; do B0=k${k}edyswjasym_r0; st=$B0$(sfx $s); w=$(wqd $st); [ -n "$w" ] || { log "no $st"; continue; }
    sp=$D5/h-${B0}_sp$(sfx $s); [ -d $sp/wq ] && { lock; ev ag-${B0}_sp$(sfx $s) $sp/wq $GA "" $(hm $sp); unlock; }
    for c in C B T; do t=${B0}_h$c$(sfx $s); d=$O/h-$t; [ -f $O/ag-$t.f32 ] && { pairboth ${B0}_sp$(sfx $s) $t; continue; }
      lock; mkdir -p $d
      [ -f $d/params.pt ] || python3 $E6/heal.py train $d --wq $w --data meta --train-meta $O/m$c --epochs 1 --ntrain 2048 --per-doc 1 >> $d.log 2>&1 || { log "heal FAIL $t"; unlock; continue; }
      [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $w $d $M >> $d.log 2>&1
      evboth $t $d/wq $(hm $d); unlock; pairboth ${B0}_sp$(sfx $s) $t; done; done
    for c in C B T; do python3 $X3/summ.py $O ${B0}_sp ${B0}_h$c ag-${B0}_sp ag-${B0}_h$c | tee -a $O/result.txt; done; done; fi
log "D1108 DONE"
