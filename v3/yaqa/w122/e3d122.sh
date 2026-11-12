#!/bin/bash
# e3d122.sh STEP ... -- w122 E3 (d) on ds8 (RULES.md): V2 3.00 / 4.00 trunks, U (R16s003 uniform) vs S-slab heads spliced as
# bf16-dequantized rows, scored on the fresh two-part slice golden-hd122. Restartable; outputs .part then renamed.
#   e3d122.sh slice               golden-hd122-x (pickx, xc8k-b2-c1 in $W/in) and golden-hd122-c (golden_pick, 37 docs)
#   e3d122.sh trunks              V2 4.00 from HF (3.00 = $Q/mixes), bf16 container (rad-convert), "trunk + bf16 head" bases
#   e3d122.sh mix ARM             one arm's container (ARM = t3-U3 t3-S3 t3-U4 t3-S4 t4-U4 t4-S4 t4-U5 t4-S5)
#   e3d122.sh score CARD ARM      both parts (t3-own / t4-own = the V2 containers; t3-bf16 / t4-bf16 = trunk + bf16 head, report only)
#   e3d122.sh read                readout.py pair S -> U per pair, rule verdicts
set -u
Q=/data/q27 W=/data/q27/heads LK=${LK:-/data/q27/locks}
source $Q/tools/env-ds8.sh
C=$W/w119/code; C2=$W/code; E=$W/e3d; S=$W/slices; CK=$Q/models/qwen38-27b-bf16
I=$Q/rad/install H=$Q/rad/home; HF=Hero422/qwen3.8-27b-exl3-v2-rad
V3=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad V4=$E/trunks/qwen3.8-27b-exl3-v2-4.00bpw.rad BF=$E/trunks/qwen38-27b-bf16.rad
mkdir -p $E/trunks $E/mix $E/scores $S
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $E/e3d122.log; }
die() { log "FAIL $*"; exit 1; }
excl() { ( exec 9>$LK/conv.lock; flock 9; exec 7>$LK/ram.lock; flock 7; "$@" ); }
step=$1; shift
case $step in
slice)
  GX=$S/golden-hd122-x GC=$S/golden-hd122-c
  if [ ! -f $GX/kld.json ]; then
    ex=(); for f in $W/in/excl/*.json; do ex+=(--exclude-orig $f); done
    rm -rf $GX.part
    $PY -I $C2/eval/slices/pickx.py $W/in/xc8k-b2-c1 $W/in/xc8k-b2-c1.jsonl $W/in/xc8k-b2-c1/corpus.meta.jsonl 130 11222 \
      $GX.part $GX.jsonl "${ex[@]}" --cap-cat agentic,tools,long,reasoning:2048 2> $GX.pick.log && mv $GX.part $GX && mv $GX.part.docs $GX.docs \
      || die "pickx ($GX.pick.log)"
    log "x part: $(head -1 $GX.pick.log) | $(tail -1 $GX.pick.log)"
  fi
  if [ ! -f $GC/kld.json ]; then
    rm -rf $GC.part
    $PY -I $C2/eval/golden_pick.py --maxtok 4096 $Q/challenge/golden/calib $Q/challenge/corpus/calib.jsonl $GC.part $GC.jsonl \
      $(cat $W/in/hd122-c.docs) > $GC.pick.log 2>&1 && mv $GC.part $GC || die "golden_pick ($GC.pick.log)"
    log "c part: $(tail -1 $GC.pick.log)"
  fi
  ;;
trunks)
  [ -f $V4 ] || { hf download $HF $(basename $V4) --local-dir $E/trunks > $E/dl4.log 2>&1; [ -f $V4 ] || die "V2 4.00 download"; log "V2 4.00 downloaded"; }
  [ -f $BF ] || { excl env RADIANCE_HOME=$H nice $I/bin/rad-convert $CK -o $BF.part > $E/convert.log 2>&1 && mv $BF.part $BF || die "rad-convert"; log "bf16 container built"; }
  for t in 3 4; do b=$E/trunks/t$t-bf16head.rad; [ -f $b ] && continue
    base=$([ $t = 3 ] && echo $V3 || echo $V4); echo "output.weight $BF" > $E/trunks/t$t.mix
    excl $Q/tools/radmix $b.part $base $E/trunks/t$t.mix > $E/trunks/t$t-mix.log 2>&1 && mv $b.part $b || die "radmix t$t"
    log "t$t + bf16 head base built"
  done
  ;;
mix)
  arm=$1; t=${arm:1:1}; f=${arm#*-}; fam=${f:0:1}; B=${f:1}; M=$E/mix/$arm.rad
  [ -f $M ] && exit 0
  if [ $fam = U ]; then perm=$W/w119/perms/R16.npy qd=$W/q/R16s003 plan=$E/plans-u.json pn=u$B
  else perm=$W/w119/perms/S.npy qd=$W/q/S
    if [ $B = 3 ]; then plan=$E/plans-u.json pn=u3; else plan=$W/w119/slab/plan.json pn=s$B-3; fi; fi
  [ -f $E/plans-u.json ] || $PY -c "import json; json.dump({'plans': {f'u{k}': {'K_blocks': [float(k)] * 1940, 'avg_bits': float(k)} for k in (3, 4, 5)}}, open('$E/plans-u.json', 'w'))"
  excl bash -c "cp $E/trunks/t$t-bf16head.rad $M.part && nice $PY -I $C/v3/yaqa/headslab.py build $plan $pn $perm $qd $M.part" > $E/mix/$arm.log 2>&1 \
    && mv $M.part $M || { rm -f $M.part; die "mix $arm ($E/mix/$arm.log)"; }
  log "mix $arm ($fam $pn on t$t): $(tail -1 $E/mix/$arm.log | cut -c1-80)"
  ;;
score)
  card=$1 arm=$2
  case $arm in t3-own) M=$V3 ;; t4-own) M=$V4 ;; t?-bf16) M=$E/trunks/${arm%-bf16}-bf16head.rad ;; *) M=$E/mix/$arm.rad ;; esac   # bf16: report only
  [ -f $M ] || die "score $arm: no container"
  for p in x c; do o=$E/scores/$arm-$p; [ -f $o.json.rows ] && continue
    ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7
      bash $Q/tools/score.sh $I $H $M $S/golden-hd122-$p $o $card ) >> $E/e3d122.log
    [ -f $o.json.rows ] || die "score $arm-$p ($o.log)"
  done
  ;;
read)
  mkdir -p $E/read
  for pr in t3:3 t3:4 t4:4 t4:5; do t=${pr%:*} B=${pr#*:}
    $PY -I $C2/eval/docstat/readout.py pair $E/read/$t-B$B.json --part x $S/golden-hd122-x $E/scores/$t-S$B-x.json.rows $E/scores/$t-U$B-x.json.rows \
      --part c $S/golden-hd122-c $E/scores/$t-S$B-c.json.rows $E/scores/$t-U$B-c.json.rows > $E/read/$t-B$B.txt 2>&1 || log "read $t B$B failed"
    for a in own; do $PY -I $C2/eval/docstat/readout.py pair $E/read/$t-own-U$B.json --part x $S/golden-hd122-x $E/scores/$t-own-x.json.rows $E/scores/$t-U$B-x.json.rows \
      --part c $S/golden-hd122-c $E/scores/$t-own-c.json.rows $E/scores/$t-U$B-c.json.rows > $E/read/$t-own-U$B.txt 2>&1; done
  done
  log "read done"
  ;;
esac
