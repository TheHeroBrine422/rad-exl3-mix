#!/bin/bash
# run.sh -- w06-heal (ranked.md H1, H4): frozen-code healing gates on the 0.8B, 4070, one GPU. README.md.
# Idempotent: a heal resumes from its state.pt, a folded arm is skipped once its .f32 exists. STAGES="gate h1 h4".
E=/workspace/rad-exl3-mix/proxy2b/exp/w06-heal P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b
Y=/data/yb O=/data/yb/w06-heal G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
STAGES=${STAGES:-gate h1 h4}
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
ev() { tag=$1 wq=$2 model=${3:-$M}; [ -f $O/$tag.f32 ] && return
  log "$tag: $(python3 $P/evalq.py $model $G --wq $wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pf() { [ -f $O/$1.f32 ] && echo $O/$1.f32 || echo $Y/$1.f32; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $(pf $1) $(pf $2) | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
# heal TAG WQ [heal.py args]: train (resumable), fold, score
heal() { tag=$1 wq=$2; shift 2; d=$O/h-$tag
  [ -f $O/$tag.f32 ] && return
  [ -f $d/params.pt ] || python3 $E/heal.py train $d --wq $wq "$@" >> $d.log 2>&1 || { log "$tag: heal FAIL (see $d.log)"; return 1; }
  log "$tag: $(grep '^HEAL DONE' $d.log | tail -1)"
  fold $tag $d/params.pt $wq; }
fold() { tag=$1 par=$2 wq=$3; shift 3; d=$O/h-$tag
  [ -f $d/wq/.done ] || python3 $E/fold.py $par $wq $d $M "$@" >> $d.log 2>&1 || { log "$tag: fold FAIL"; return 1; }
  ev $tag $d/wq $([ -f $d/model/.done ] && echo $d/model || echo $M); }

if [[ " $STAGES " == *" gate "* ]] && [ ! -f $O/gate.ok ]; then
  python3 $E/heal.py train $O/gate --wq $Y/q-k3base/wq --gate > $O/gate.log 2>&1
  grep -q "^GATE OK" $O/gate.log || { log "GATE FAIL (see $O/gate.log)"; exit 1; }
  log "$(grep -E '^gate identity|^GATE OK' $O/gate.log | tr '\n' ' ')"; touch $O/gate.ok
fi

if [[ " $STAGES " == *" h1 "* ]]; then
  fold k3jit - $Y/q-k3base/wq --jitter 0.003; pair k3base k3jit                  # re-rounding control
  for k in 3 2; do                                                               # H1(c): gradient precision
    [ -f $O/gc-k$k/gradcos.txt ] || python3 $E/heal.py gradcos $O/gc-k$k --wq $Y/q-k${k}base/wq > $O/gc-k$k.log 2>&1
    log "gradcos K$k: $(tail -1 $O/gc-k$k/gradcos.txt)"
  done
  for k in 3 2; do                                                               # H1(a), H1(b)
    heal k${k}all $Y/q-k${k}base/wq
    heal k${k}last $Y/q-k${k}base/wq --layers 18-23
    heal k${k}orc $Y/q-k${k}base/wq --data oracle --epochs 6 --lr 3e-3 --patience 99
    pair k${k}base k${k}all; pair k${k}base k${k}last; pair k${k}all k${k}last; pair k${k}base k${k}orc
  done
  heal k3dall $Y/q-k3bdiag/wq; pair k3bdiag k3dall                              # stacks on diagonal H_out?
fi

if [[ " $STAGES " == *" h4 "* ]]; then
  for m in F D; do mp=$([ $m = F ] && echo family || echo depth); x=$O/mix-$m
    [ -f $x/xfer.pt ] || python3 $E/mkmix.py $mp $x $Y/q-k2base/wq $Y/q-k3base/wq $O/h-k2all/params.pt $O/h-k3all/params.pt | tee -a $O/result.txt
    ev mix$m $x/wq                                                               # unhealed mix
    fold mix${m}x $x/xfer.pt $x/wq                                               # level library, transferred
    heal mix${m}h $x/wq                                                          # mix-specific, same budget
    heal mix${m}w $x/wq --init $x/xfer.pt --max-steps 128                        # library + 1/4-budget polish
    pair mix$m mix${m}x; pair mix$m mix${m}h; pair mix$m mix${m}w; pair mix${m}h mix${m}x
  done
fi
python3 $E/summ.py $O $G $Y | tee -a $O/result.txt
log W06 DONE
