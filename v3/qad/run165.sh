#!/bin/bash
# run165.sh STEP... -- the #165 pilot on ds8 (rules: v3/qad/RULES.md, 163ef05). Every step skips what is done; rerun
# the same line after any failure or kill. Steps:
#   qadA qadC qadD   QAD arms A on $QA, C on $QC, D on $QA (D waits for d0's final-asym); one card each holds the
#                    whole int8 student (~25 GiB with the bf16 head; amendment 1), a comma list splits the layers
#   qadB             arm B (student + bf16 teacher, ~70 GiB) on the 3 cards $BC
#   gate:ARM         qad.py --gate for ARM on its cards (memory / timing before the runs)
#   d0               asym writers per mix on card $DC: sibling moments (1M qad tokens) -> denoise (lam 1, rho 0) ->
#                    no-op gate (2 writer jobs from the level's own specs = the level, bit for bit) -> the 128 writer
#                    jobs with hin / wpre -> splice into the level (pre-C9) -> C9 on the asym writers (--dn) -> final-asym
#   ev:TAG           pack + mix (over V2 3.00) + score on the c / x slices, card $EC: TAG = control A B C D0 D
#   cmp              paired readouts (healeval.py cmp) -> $W/result.json
# Amendment 2 (lead notes 11-13): R16 head K4 (u4) arms A16 (head frozen) / H (head r / c trained), W (8,192-token
# windows, same tokens), L8 / L16 (rank-8 / 16 adapters on the base X = H if H beats A16, else A16; torch-path scored),
# long-context checks:
#   qadA16 qadH qadW qadL8 qadL16   the runs (cards QA16 QH WC L8C L16C)
#   h16ctl / h16H    head export (rxq.deq W, f16) + a donor container (copy of control's trunk pack, output.weight
#                    overwritten by headslab.py build: bf16 rows in vocab order)
#   ev:control16 ev:A16 ev:H ev:W   as ev (A16 / H / control16 with their donor's head)
#   evlong:TAG       TAG's mix on agent-verify32 (32k chunks, --max-len 32768) -> result-long.json; TAG = control A W
#   lbase            writes $W/LBASE (H or A16) by the B / C rule on cmp H / A16
#   tscore:RUN       torch-path scores of RUN's student on c / x (card TSC)
#   cmp2 / pos       amendment 2's readouts, posci.py by bin for every judged arm
#   all-lanes        four lanes in parallel, each step taking its card locks (ascending order, so no deadlock):
#                    (qadA, qadB, qadD) (qadC) (d0) (ev:control ev:A ev:C ev:B ev:D0 ev:D cmp); 4 cards
# Never starts while the trunk ladder runs (an rxrun.run process; FORCE=1 overrides). Locks /data/q27/locks: card locks
# for every GPU step (QAD holds its cards for the whole arm), packs / mixes conv + ram exclusive (healeval). QAD does
# not take ram.lock (ds8: 235 GB; a held shared lock would starve the packs' exclusive one for hours).
set -u
Q=/data/q27; W=$Q/w125; CODE=$W/code; STAGE=$Q/stage/code
PY=$Q/venv/bin/python
if [ -d $W/k2.5 ]; then LADDER=${LADDER:-$W/k2.5} ENVF=${ENVF:-$Q/tools/env-box2.sh} EC=${EC:-0}   # box2 (amendment 1)
else LADDER=${LADDER:-$Q/ladder/q27v3-144dd6e1-k2.5-i8}; fi   # the level's dir (final, level, specs, final.recipe)
LEVEL=$LADDER/final; LEVEL0=$LADDER/level
MODEL=$Q/models/qwen38-27b-bf16; GOLD=$Q/challenge/golden/calib; HESS=$Q/challenge/hessians/calib
POOLS=$CODE/eval/pools/pools-v3.json
ENVF=${ENVF:-$Q/tools/env-ds8.sh}   # the rxq job env (box2: env-box2.sh)
SELF=$(readlink -f "$0")
QA=${QA:-0} QC=${QC:-1} BC=${BC:-0,1,3} DC=${DC:-2} EC=${EC:-3}     # box2's 4 cards (ds8: QA=4 QC=5 BC=4,5,7 DC=6 EC=7)
D=$W/d0; R=$W/runs; E=$W/eval; LK=$Q/locks
SLICES=(--slice c=$W/slices/golden-verify27-w114 --slice x=$W/slices/golden-hd122-x)
HK=$Q/w119/q/R16s003/K4; PERM=$Q/w119/perms/R16.npy; HD=$W/h16
HEADA=(--head $HK --head-perm $PERM)
QA16=${QA16:-0} QH=${QH:-1} WC=${WC:-2,3} L8C=${L8C:-1} L16C=${L16C:-2,3} TSC=${TSC:-0}
mkdir -p $R $E $D
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run165.log; }
die() { log "FAIL $*"; exit 1; }
cards() {   # cards "4,5" CMD... : CMD under every card's lock
  local cs=$1; shift; local pre=()
  for c in ${cs//,/ }; do pre+=(flock $LK/card$c.lock); done
  "${pre[@]}" "$@"
}
[ -n "${FORCE:-}" ] || ! pgrep -f "[r]xrun.run " > /dev/null || die "the trunk ladder (rxrun.run) is running: not starting"
[ -f $W/STOP ] && die "STOP file"

qad() {     # qad ARM LEVEL NAME CARDS [extra qad.py args]
  local arm=$1 lv=$2 name=$3 cs=$4 out=$R/$3; shift 4
  [ -f $out/params.pt ] && { log "qad $name done"; return 0; }
  local n=$(echo ${cs//,/ } | wc -w); local devs=$(seq -s, -f "cuda:%g" 0 $((n - 1)))
  log "qad $name: arm $arm, level $lv, cards $cs"
  cards $cs env ROCR_VISIBLE_DEVICES=$cs PYTHONPATH=$W/pylib-fla QAD_PIPELINE=$STAGE/pipeline TRITON_DEFAULT_FP_FUSION=1 \
    OMP_NUM_THREADS=8 nice -n 5 $PY $CODE/v3/qad/qad.py $out --model $MODEL --level $lv --golden $GOLD --pools $POOLS \
    --devs $devs --arm $arm --ntrain 1024 --accum 8 --eval-every 16 --patience 3 --monitor 32 --len 2048 --lr 1e-3 \
    --warmup 8 --trust 0.3 --seed 0 "$@" >> $R/$name.log 2>&1 || die "qad $name rc $? ($R/$name.log)"
  [ -n "${GATE:-}" ] && { log "gate $name: $(grep -h 'gate' $out/run.log | tail -n 4 | cut -c10- | tr '\n' ' ')"; return 0; }
  [ -f $out/params.pt ] || die "qad $name: no params.pt"
  log "qad $name: $(grep 'QAD DONE' $out/run.log | tail -n 1 | cut -c10-)"
}

d0() {
  [ -f $D/final-asym/.done ] && { log "d0 done"; return 0; }
  ( source $ENVF; export POOLS
    if [ $(ls $D/mom/.done 2>/dev/null | wc -l) -lt 64 ]; then
      log "d0 moments (card $DC)"; t0=$(date +%s)
      cards $DC env DEV=cuda ROCR_VISIBLE_DEVICES=$DC nice -n 5 $PY -m rxq.sibmom $MODEL $GOLD $D/mom --level $LEVEL \
        --codebook i8 --tokens 1000000 --wins moments >> $D/mom.log 2>&1 || die "sibmom rc $? ($D/mom.log)"
      [ $(ls $D/mom/.done | wc -l) -ge 64 ] || die "sibmom: not 64 layers"
      log "d0 moments done in $(( ($(date +%s) - t0) / 60 )) min"
    fi
    if [ ! -f $D/dn/.done ]; then
      log "d0 denoise"; t0=$(date +%s)
      # rxq.asym + asym-w125.patch (w78's blocked Cholesky solve on the lean path; resumable): the staged rxq.asym's
      # torch.linalg.solve at k 17408 f64 takes > 15 min a down group on the R9700 (rocBLAS's degraded TRSM)
      cards $DC env DEV=cuda ROCR_VISIBLE_DEVICES=$DC nice -n 5 $PY $W/asymfix/asym.py denoise $HESS $D/mom $D/dn --lam 1 \
        --rho 0 --ckpt $MODEL >> $D/dn.log 2>&1 || die "denoise rc $? ($D/dn.log)"
      log "d0 denoise done in $(( ($(date +%s) - t0) / 60 )) min"
    fi
    if [ ! -f $D/noop.ok ]; then
      log "d0 no-op gate"
      cards $DC env ROCR_VISIBLE_DEVICES=$DC $PY $CODE/v3/qad/d0jobs.py noop $LADDER/specs $D/noop L00-g1 L03-g1 >> $D/noop.log 2>&1 || die "noop jobs"
      for s in L00-g1 L03-g1; do
        $PY -m rxq.pairs same $LEVEL0 $D/noop/$s >> $D/noop.log 2>&1 || die "no-op gate $s: the job does not reproduce the level ($D/noop.log)"
      done
      touch $D/noop.ok; log "d0 no-op gate PASS (L00 out_proj, L03 o_proj bit-identical to the level)"
    fi
    if [ ! -f $D/writers.ok ]; then
      log "d0 writer jobs"; t0=$(date +%s)
      cards $DC env ROCR_VISIBLE_DEVICES=$DC nice -n 5 $PY $CODE/v3/qad/d0jobs.py asym $LADDER/specs $D/out --hin $D/dn/hin \
        --wpre $D/dn/wpre >> $D/jobs.log 2>&1 || die "writer jobs ($D/jobs.log)"
      mkdir -p $D/writers; for f in $D/out/*/t-*.safetensors; do ln -sf $f $D/writers/; done
      [ $(ls $D/writers | wc -l) = 128 ] || die "writers: $(ls $D/writers | wc -l) tensors, not 128"
      touch $D/writers.ok; log "d0 writer jobs done in $(( ($(date +%s) - t0) / 60 )) min"
    fi
    if [ ! -f $D/level-asym/spliced.json ]; then
      $PY -m rxq.pairs splice $LEVEL0 $D/writers $D/level-asym --arm attn_output,ssm_out,ffn_down >> $D/splice.log 2>&1 || die "splice"
    fi
    if [ ! -f $D/c9dn/.done ]; then
      log "d0 C9 on the asym writers"
      cards $DC env DEV=cuda ROCR_VISIBLE_DEVICES=$DC nice -n 5 $PY -m rxq.post c9 $D/level-asym $D/c9dn --model $MODEL \
        --hess $HESS --dn $D/dn >> $D/c9.log 2>&1 || die "c9 ($D/c9.log)"
    fi
    $PY -m rxq.post apply $D/level-asym $D/c9dn $D/final-asym >> $D/c9.log 2>&1 || die "apply c9"
    [ -f $D/final-asym/.done ] || die "final-asym not done"
    log "d0 final-asym ready"
  ) || exit 1
}

h16() {     # h16 NAME [RUN]: $HD/NAME/K4 + $HD/donor-NAME.rad
  local n=$1 run=${2:-}; local o=$HD/donor-$n.rad
  [ -f $o ] && { log "h16 $n done"; return 0; }
  [ -n "$run" ] && waitfor $run/params.pt
  mkdir -p $HD
  [ -f $HD/plan-u4.json ] || $PY -c "import json; json.dump({'plans': {'u4': {'K_blocks': [4] * 1940}}}, open('$HD/plan-u4.json', 'w'))"
  local hx=(--ref $HK/wq/lm_head.safetensors); [ -n "$run" ] && hx=(--run $run)
  ( cd $W; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 nice -n 10 $PY $CODE/v3/qad/headexport.py $HD/$n $HK --dev cpu "${hx[@]}" \
      >> $HD/$n.log 2>&1 ) || die "headexport $n"
  ( exec 9>$LK/conv.lock; flock 9; exec 7>$LK/ram.lock; flock -s 7
    rm -f $o.part; cp $E/control/trunk.rad $o.part && nice -n 10 $PY -I $CODE/v3/yaqa/headslab.py build $HD/plan-u4.json u4 $PERM $HD/$n $o.part >> $HD/$n.log 2>&1 \
      && mv $o.part $o ) || { rm -f $o.part; die "donor $n ($HD/$n.log)"; }
  log "h16 $n: $o"
}

tscore() {  # tscore RUN: torch-path scores on c / x with RUN's own student settings
  local run=$R/$1; waitfor $run/params.pt
  [ -f $run/tscore-x.npz ] && { log "tscore $1 done"; return 0; }
  local ex=($($PY -c "
import torch; a = torch.load('$run/params.pt', weights_only=False)['meta']['args']
print(' '.join((['--head', a['head']] if a.get('head') else []) + (['--head-perm', a['head_perm']] if a.get('head_perm') else [])
               + (['--head-train'] if a.get('head_train') else []) + (['--lora', str(a['lora'])] if a.get('lora') else [])))"))
  log "tscore $1 (card $TSC): ${ex[*]}"
  cards $TSC env ROCR_VISIBLE_DEVICES=$TSC PYTHONPATH=$W/pylib-fla QAD_PIPELINE=$STAGE/pipeline TRITON_DEFAULT_FP_FUSION=1 \
    OMP_NUM_THREADS=8 nice -n 5 $PY $CODE/v3/qad/qad.py $run --model $MODEL --level $($PY -c "import torch; print(torch.load('$run/params.pt', weights_only=False)['meta']['args']['level'])") \
    --golden $GOLD --pools $POOLS --devs cuda:0 --init $run/params.pt "${ex[@]}" --score c=$W/slices/golden-verify27-w114 \
    --score x=$W/slices/golden-hd122-x >> $R/$1.tscore.log 2>&1 || die "tscore $1"
  log "tscore $1: $(grep -h 'score [cx]:' $run/run.log | tail -n 2 | cut -c10- | tr '\n' ' ')"
}

waitfor() { while [ ! -f $1 ]; do [ -f $W/STOP ] && die STOP; sleep 60; done; }

ev() {      # ev TAG (waits for its input)
  local t=$1 lv=$LEVEL sc=() rc=(--recipe $LADDER/final.recipe)
  [ -f $E/$t/result.json ] && { log "ev $t done"; return 0; }
  case $t in
    control) ;;
    A|B|C) sc=(--scales $R/$t); waitfor $R/$t/params.pt;;
    D0) lv=$D/final-asym; waitfor $D/final-asym/.done;;
    D) lv=$D/final-asym; sc=(--scales $R/D); waitfor $R/D/params.pt;;
    W) sc=(--scales $R/W); waitfor $R/W/params.pt;;
    control16) waitfor $HD/donor-ctl.rad; sc=(--head-donor $HD/donor-ctl.rad);;
    A16) waitfor $R/A16/params.pt; waitfor $HD/donor-ctl.rad; sc=(--scales $R/A16 --head-donor $HD/donor-ctl.rad);;
    H) waitfor $HD/donor-H.rad; sc=(--scales $R/H --head-donor $HD/donor-H.rad);;
    *) die "ev $t?";;
  esac
  log "ev $t: pack + mix + score (card $EC)"
  ( source $ENVF
    nice -n 5 $PY $CODE/v3/qad/healeval.py $t $lv $E --host $Q/multi/host.json "${sc[@]}" "${rc[@]}" "${SLICES[@]}" \
      --card $EC --card-lock $LK/card$EC.lock >> $E/$t.log 2>&1 ) || die "ev $t ($E/$t.log)"
  log "ev $t: $($PY -c "import json; r = json.load(open('$E/$t/result.json')); print(', '.join(f\"{k} {v['kld']:.5f} ({v['positions']} pos)\" for k, v in r['slices'].items()))")"
}

evlong() {  # evlong TAG: TAG's finished mix on the 32k verify trace chunks
  local t=$1
  [ -f $E/$t/result-long.json ] && { log "evlong $t done"; return 0; }
  waitfor $E/$t/result.json
  local sc=(); [ $t != control ] && sc=(--scales $R/$t)
  log "evlong $t (card $EC)"
  ( source $ENVF
    nice -n 5 $PY $CODE/v3/qad/healeval.py $t $LEVEL $E --host $Q/multi/host.json "${sc[@]}" --recipe $LADDER/final.recipe \
      --slice l=$W/slices/golden-agent-verify32 --max-len 32768 --result result-long.json --card $EC --card-lock $LK/card$EC.lock >> $E/$t.long.log 2>&1 ) || die "evlong $t"
  log "evlong $t: $($PY -c "import json; r = json.load(open('$E/$t/result-long.json')); print(', '.join(f\"{k} {v['kld']:.5f} ({v['positions']} pos)\" for k, v in r['slices'].items()))")"
}

cmp1() {    # cmp1 A B [RESULT]: one paired readout, logged
  local a=$1 b=$2 res=${3:-result.json} sfx=${3:+-long}
  [ -f $E/$a/$res ] && [ -f $E/$b/$res ] || return 0
  ( source $ENVF; $PY $CODE/v3/qad/healeval.py cmp $E $a $b $res > $E/cmp-$a-$b$sfx.json ) || die "cmp $a $b"
  log "cmp $b / $a$sfx: $($PY -c "import json; r = json.load(open('$E/cmp-$a-$b$sfx.json'))['slices']; print('; '.join(f\"{k} {v['ratio']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] better {v['docs_better']} worse {v['docs_worse']}\" for k, v in r.items()))")"
}

cmp2() {
  for p in control16:A16 control16:H A16:H control:W A:W control:A16; do cmp1 ${p%:*} ${p#*:}; done
  for p in control:A control:W A:W; do cmp1 ${p%:*} ${p#*:} result-long.json; done
}

lbase() {   # the B / C rule on H vs A16: c CI below 1 and x below 1 -> H, else A16
  [ -f $W/LBASE ] && { log "lbase $(cat $W/LBASE)"; return 0; }
  waitfor $E/H/result.json; waitfor $E/A16/result.json; cmp1 A16 H
  $PY -c "
import json; r = json.load(open('$E/cmp-A16-H.json'))['slices']
print('H' if r['c']['ci95'][1] < 1 and r['x']['ratio'] < 1 else 'A16')" > $W/LBASE.part && mv $W/LBASE.part $W/LBASE
  log "lbase: $(cat $W/LBASE) (H vs A16: the B / C rule)"
}

qadL() {    # qadL RANK CARDS
  waitfor $W/LBASE; local hx=(); [ "$(cat $W/LBASE)" = H ] && hx=(--head-train)
  qad A $LEVEL L$1 $2 "${HEADA[@]}" "${hx[@]}" --lora $1 --lr-lora 1e-4 "${@:3}"
}

pos() {     # posci.py for every judged arm vs its control, c / x (+ the long slice)
  mkdir -p $E/pos
  for p in control:A control:B control:C control:D0 control:D control16:A16 control16:H A16:H control:W A:W; do
    a=${p%:*} b=${p#*:}; [ -f $E/$a/result.json ] && [ -f $E/$b/result.json ] || continue
    for sl in c x; do
      $PY $CODE/v3/qad/posci.py $W/slices/$([ $sl = c ] && echo golden-verify27-w114 || echo golden-hd122-x) $E/$a/score-$sl.json.rows \
        $E/$b/score-$sl.json.rows --label "$b/$a $sl" --json $E/pos/$a-$b-$sl.json
    done
  done | tee $E/pos/pos.txt
  for p in control:A control:W A:W; do
    a=${p%:*} b=${p#*:}; [ -f $E/$a/result-long.json ] && [ -f $E/$b/result-long.json ] || continue
    $PY $CODE/v3/qad/posci.py $W/slices/golden-agent-verify32 $E/$a/score-l.json.rows $E/$b/score-l.json.rows --label "$b/$a l" \
      --json $E/pos/$a-$b-l.json
  done | tee -a $E/pos/pos.txt
}

lcmp() {    # arm L vs its base X on the torch rows: raw + size-adjusted ratios, position bins
  waitfor $W/LBASE; local x=$(cat $W/LBASE)
  for r in 8 16; do
    [ -f $R/L$r/tscore-x.npz ] && [ -f $R/$x/tscore-x.npz ] || continue
    for sl in c x; do
      $PY $CODE/v3/qad/lcmp.py $R/$x $R/L$r $sl --lora-run $R/L$r > $E/lcmp-L$r-$sl.json || die "lcmp L$r $sl"
      log "lcmp L$r / $x $sl: $($PY -c "import json; r = json.load(open('$E/lcmp-L$r-$sl.json')); print(f\"raw {r['ratio']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] better {r['docs_better']} worse {r['docs_worse']}; \" + '; '.join(f\"{t} +{r['size_'+t]['extra_bpw']:.4f} bpw adj {r['size_'+t]['adj_ratio']:.4f} [{r['size_'+t]['adj_ci95'][0]:.4f}, {r['size_'+t]['adj_ci95'][1]:.4f}]\" for t in ('bf16', 'int8')))")"
      $PY $CODE/v3/qad/posci.py $W/slices/$([ $sl = c ] && echo golden-verify27-w114 || echo golden-hd122-x) $R/$x/tscore-$sl.npz \
        $R/L$r/tscore-$sl.npz --label "L$r/$x $sl" --json $E/pos/$x-L$r-$sl.json | tee -a $E/pos/pos.txt
    done
  done
}

cmpall() {
  ( source $ENVF
    for p in control:A control:B control:C control:D0 control:D A:B A:C A:D; do
      a=${p%:*} b=${p#*:}
      [ -f $E/$a/result.json ] && [ -f $E/$b/result.json ] || continue
      $PY $CODE/v3/qad/healeval.py cmp $E $a $b > $E/cmp-$a-$b.json || die "cmp $a $b"
      log "cmp $b / $a: $($PY -c "import json; r = json.load(open('$E/cmp-$a-$b.json'))['slices']; print('; '.join(f\"{k} {v['ratio']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] better {v['docs_better']} worse {v['docs_worse']}\" for k, v in r.items()))")"
    done )
}

for s in "$@"; do
  case $s in
    qadA) qad A $LEVEL A $QA;;
    qadC) qad C $LEVEL C $QC;;
    qadD) waitfor $D/final-asym/.done; qad A $D/final-asym D $QA;;
    qadA16) qad A $LEVEL A16 $QA16 "${HEADA[@]}";;
    qadH) qad A $LEVEL H $QH "${HEADA[@]}" --head-train;;
    qadW) qad A $LEVEL W $WC --len 8192 --ntrain 668 --accum 5 --conv shift;;
    qadL8) qadL 8 $L8C;;
    qadL16) qadL 16 $L16C --conv shift;;   # amendment 4: 2-card runs stall in MIOpen's conv
    gate:A16) GATE=1 qad A $LEVEL gateA16 $QA16 "${HEADA[@]}" --head-train --gate;;
    gate:W) GATE=1 qad A $LEVEL gateW $WC --len 8192 --ntrain 668 --accum 5 --gate;;
    gate:L16) GATE=1 qad A $LEVEL gateL16 $L16C "${HEADA[@]}" --head-train --lora 16 --gate;;
    h16ctl) h16 ctl;;
    h16H) h16 H $R/H;;
    evlong:*) evlong ${s#evlong:};;
    lbase) lbase;;
    tscore:base) waitfor $W/LBASE; tscore $(cat $W/LBASE);;
    tscore:*) tscore ${s#tscore:};;
    lcmp) lcmp;;
    lanes2)     # amendment 2's lanes: (card 0) A16 then every radiance eval + torch scoring; (card 1) H, head donor, L8;
                # (cards 2-3) W, L16
      ( bash $SELF qadA16 ev:A16 ev:H lbase tscore:base ev:control16 evlong:control evlong:A ev:W evlong:W cmp2 pos \
          tscore:L8 tscore:L16 lcmp ) & p1=$!
      ( bash $SELF qadH h16H qadL8 ) & p2=$!
      ( bash $SELF qadW qadL16 ) & p3=$!
      for p in $p1 $p2 $p3; do wait $p || die "a lane failed (see the log)"; done
      log "LANES2 DONE";;
    cmp2) cmp2;;
    pos) pos;;
    gate:A) GATE=1 qad A $LEVEL gateA $QA --gate;;
    gate:B) GATE=1 qad B $LEVEL gateB $BC --gate;;
    qadB) qad B $LEVEL B $BC;;
    d0) d0;;
    ev:*) ev ${s#ev:};;
    cmp) cmpall;;
    all-lanes)
      ( bash $SELF qadA qadB qadD ) & p1=$!
      ( bash $SELF qadC ) & p2=$!
      ( bash $SELF d0 ) & p3=$!
      ( bash $SELF ev:control ev:A ev:C ev:B ev:D0 ev:D cmp ) & p4=$!
      for p in $p1 $p2 $p3 $p4; do wait $p || die "a lane failed (see the log)"; done
      log "ALL LANES DONE";;
    *) die "step $s?";;
  esac
done
