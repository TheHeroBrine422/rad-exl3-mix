#!/bin/bash
# run.sh -- w10-longpos (ranked.md I4): does the 0.8B diagonal-H_out gain hold at context positions > 2k? Builds a long
# 0.8B golden (/data/g08-long1: the 22 search-pool documents >= 8k tokens, the two agent traces cut at 64k; ~420k rows,
# 236k scored) and re-scores the EXISTING quants on it (fixed codes: no new quants unless the rule asks for seeds):
# base / edy (Σdy²) / eB (Sketch B) x seeds 0-2 x K3 / K2, every other base seed present, and w10-hout-attrib's family
# arms when they exist (family x position). 4070, one GPU, idempotent and resumable.
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/w10-longpos M=/data/models/qwen35-08b G=/data/g08-search1 GL=/data/g08-long1
Y=/data/yb O=/data/yb/w10-longpos A=/data/yb/w10-hout-attrib C=/data/yb/c18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
while pgrep -f "quantize/[o]urq.py|proxy2b/[e]valq.py|proxy2b/[m]kgold.py" > /dev/null; do sleep 60; done
cd /tmp
# 1. the golden (frozen document list glong-docs.txt; if the 64k traces do not fit the 7.2 GB cap: 32k, logged)
if [ ! -f $GL/kld.json ]; then
  python3 $P/mkgold.py $M /data/challenge/calib-meta $GL $(cat $X/glong-docs.txt) > $O/mkgold.log 2>&1 \
    && log "golden $GL: $(python3 -c "import json; m=json.load(open('$GL/kld.json')); print(len(m['docs']), 'docs', m['rows'], 'rows')") (traces at 64k)" \
    || { log "64k golden failed ($(tail -1 $O/mkgold.log | cut -c1-120)): traces at 32k"; rm -rf $GL
         python3 $P/mkgold.py $M /data/challenge/calib-meta $GL $(cat $X/glong-docs-c32k.txt) > $O/mkgold.log 2>&1 \
           || { log "GOLDEN FAIL"; rm -rf $GL; exit 1; }; log "golden $GL built with 32k traces"; }
fi
wqdir() { for d in $Y $A $O; do [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
ev() { tag=$1; [ -f $O/$tag.glong.f32 ] && return; w=$(wqdir $tag); [ -z "$w" ] && return
  log "$tag glong: $(python3 $P/evalq.py $M $GL --wq $w --out $O/$tag.glong.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"
  [ -f $O/$tag.glong.f32 ] || log "$tag: glong eval FAILED"; }
q() { tag=$1; k=$2; shift 2   # a new quant (only for the extra seeds the rule may ask for), scored on both goldens
  if [ -z "$(wqdir $tag)" ]; then $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1; fi
  [ -f $Y/$tag.f32 ] || [ -f $O/$tag.f32 ] || log "$tag: $(python3 $P/evalq.py $M $G --wq $(wqdir $tag) --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"
  ev $tag; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
# 2. existing quants on the long golden (~1 min each)
for k in 3 2; do for s in 0 1 2 3 4 5; do
  for a in edy eB h-gdn_in h-attn_in h-mlp_in h-writers h-d0 h-d1 h-d2; do
    [ -n "$(wqdir k${k}$a$(sfx $s))" ] && { ev k${k}base$(sfx $s); ev k${k}$a$(sfx $s); }   # an arm and its seed's base
  done; done; done
# 3. extra seeds if the rule says INCONCLUSIVE (pre-registered: edy seeds 3-4, their base seeds exist after yaqaFact12)
python3 $X/posgain.py $GL $O $Y,$A,$O --need-seeds > $O/need.txt
if [ -s $O/need.txt ]; then
  log "inconclusive: extra seeds $(tr '\n' ';' < $O/need.txt)"
  while read k a s; do q k${k}base$(sfx $s) $k --seed-offset $s < /dev/null
    q k${k}edy$(sfx $s) $k --hout $C/hdiag-snap1024-dy --sigma-reg-out 0.01 --seed-offset $s < /dev/null; done < $O/need.txt
fi
python3 $X/posgain.py $GL $O $Y,$A,$O | tee -a $O/result.txt
log LONGPOS DONE
