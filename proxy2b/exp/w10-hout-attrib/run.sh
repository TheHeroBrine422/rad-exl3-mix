#!/bin/bash
# run.sh -- w10-hout-attrib (ranked.md B3): the Σdy² diagonal H_out (c18 hdiag-snap1024-dy, the 27B collector's primary
# estimator) on one family / one depth third at a time (K3, K2), and on the whole trunk at K4 / K5; 0.8B, i8, base
# settings, output damping 0.01, g08-search1. 2 seeds per arm, a 3rd / 4th only where the rule's boundary is within
# one SE (summ.py --need-seeds). 4070, one GPU, idempotent and resumable; reuses k*base* / k*edy* from /data/yb.
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/w10-hout-attrib M=/data/models/qwen35-08b G=/data/g08-search1
Y=/data/yb O=/data/yb/w10-hout-attrib C=/data/yb/c18/hdiag-snap1024-dy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
while pgrep -f "quantize/[o]urq.py|proxy2b/[e]valq.py" > /dev/null; do sleep 60; done   # the card is shared: one job at a time
[ -f $C/model.language_model.layers.23.mlp.down_proj.safetensors ] || { log "no Σdy² diagonal in $C (yaqaFact7)"; exit 1; }
# ourq with w01-failloud's check refuses partial --hout dirs unless told: these arms are partial on purpose
SP=$(grep -q -- "--side-partial" /workspace/rad-exl3-mix/quantize/ourq.py && echo --side-partial)
declare -A SUB=([gdn_in]=fam:gdn_in [attn_in]=fam:attn_in [mlp_in]=fam:mlp_in [writers]=fam:writers \
                [d0]=depth:0-7 [d1]=depth:8-15 [d2]=depth:16-23)
for n in "${!SUB[@]}"; do [ -f $O/sub/$n/.done ] || python3 $X/mksub.py $C $O/sub/$n ${SUB[$n]} | tee -a $O/result.txt; done
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
wq() { [ -d $Y/q-$1/wq ] && echo $Y/q-$1/wq || echo $O/q-$1/wq; }
q() { tag=$1; k=$2; shift 2
  [ -f $Y/$tag.f32 ] || [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
arm() { k=$1 a=$2 s=$3   # one seed of one arm, and its seed-matched base if missing
  q k${k}base$(sfx $s) $k --seed-offset $s
  case $a in
    edy) q k${k}edy$(sfx $s) $k --hout $C --sigma-reg-out 0.01 --seed-offset $s ;;
    h-*) n=${a#h-}; t=k${k}h-$n$(sfx $s)
         q $t $k --hout $O/sub/$n --sigma-reg-out 0.01 --seed-offset $s $SP
         # identity gate: outside the subset the arm must equal its base bit for bit (same seed, no H_out there)
         if [ ! -f $O/id-$t.json ]; then
           python3 $X/idcheck.py $(wq k${k}base$(sfx $s)) $(wq $t) $O/sub/$n/subset.txt > $O/id-$t.json
           log "idcheck $t: $(cat $O/id-$t.json)"
         fi ;;
  esac; }
# 1. two seeds of every subset arm at K3 / K2, and of the full diagonal at K4 / K5
for s in 0 1; do
  for k in 3 2; do for n in gdn_in attn_in mlp_in writers d0 d1 d2; do arm $k h-$n $s; done; done
  for k in 4 5; do arm $k edy $s; done
done
# a subset arm that is not identical to base outside its subset carries the whole model's rounding noise: 3 seeds
grep -l '"pass": false' $O/id-*.json > /dev/null 2>&1 && { export MINSEED=3; log "idcheck failed somewhere: MINSEED=3"; }
# 2. adaptive seeds (pre-registered: boundary within one SE -> one more seed, at most 4)
for round in 1 2; do
  python3 $X/summ.py $O,$Y --need-seeds > $O/need.txt
  [ -s $O/need.txt ] || break
  log "round $round, more seeds: $(tr '\n' ';' < $O/need.txt)"
  while read k a s; do arm $k $a $s < /dev/null; done < $O/need.txt
done
python3 $X/summ.py $O,$Y | tee -a $O/result.txt
log ATTRIB DONE
