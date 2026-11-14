#!/bin/bash
# s3level.sh LEVEL ARM [CARDS] -- RULES-177 stage 3, one level, restartable (every step skips when its output exists).
#   ARM h: the hash level q27v3-144dd6e1-k<K>-hash from box2's ladder level LEVEL (q27v3-144dd6e1-k<K>-i8): its own job
#          specs with codebook hash and the patched tree's config (env-w139.sh), run by 3 workers a card on the first free
#          card of CARDS (default "3 2 1 0"; card lock + ram.lock shared held by this script; jobs claimed by flock, so
#          several s3level.sh on several cards share one level), then assemble / post (in-job C9) / recipe as rxrun,
#          and final/stamp.json's args.codebook shimmed to i8 for the importer (original kept as stamp-hash.json).
#   ARM i: the control: the ladder level's out/ (hard links) through assemble / post / recipe with the UNPATCHED tree
#          (stage/code-old-8f73ee4, the ladder's code stamp), CPU only.
# Out /data/q27/w139/s3/<level name>/. (w139, #177)
set -u; LV=$1; ARM=$2; Q=/data/q27; H=$Q/w139; L=$Q/ladder/$LV
K=${LV#q27v3-144dd6e1-k}; K=${K%-i8}
if [ $ARM = h ]; then N=q27v3-144dd6e1-k$K-hash; source $H/tools/env-w139.sh
else N=$LV; source $Q/tools/env-box2.sh
  export PYTHONPATH=$Q/stage/code-old-8f73ee4/pipeline:$Q/stage/exl3:$Q/extbase RXQ_RXM=$Q/stage/code-old-8f73ee4; fi
D=$H/s3/$N; mkdir -p $D/specs $D/out $D/logs; exec >> $D/logs/s3level.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $N (arm $ARM, K $K)"
if [ $ARM = h ]; then
  for s in $L/specs/L*.json; do
    t=$D/specs/$(basename $s); [ -f $t ] && continue
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d['codebook']='hash'; d['level']=sys.argv[2]; d['config']=sys.argv[3]; json.dump(d,open(sys.argv[4],'w'),indent=1)" \
      $s $N $H/code/pipeline/config/v3.toml $t
  done
  todo() { for s in $D/specs/L*.json; do [ -f $D/out/$(basename $s .json)/job.json ] || echo $s; done; }
  if [ -n "$(todo)" ]; then
    while :; do for c in ${3:-3 2 1 0}; do exec 5>$Q/locks/card$c.lock; if flock -n 5; then CARD=$c; break 2; fi; exec 5>&-; done
      [ -z "$(todo)" ] && break; sleep 30; done
    if [ -n "${CARD:-}" ]; then
      exec 4>$Q/locks/ram.lock; flock -s 4; say "card $CARD: jobs"
      worker() {
        for s in $D/specs/L*.json; do n=$(basename $s .json); o=$D/out/$n
          [ -f $o/job.json ] && continue
          exec 7>$D/specs/$n.claim; flock -n 7 || { exec 7>&-; continue; }
          [ -f $o/job.json ] || { rm -rf $o; mkdir -p $o; ROCR_VISIBLE_DEVICES=$CARD nice -n 5 $PY -m rxq job $s -o $o --device 0 >> $o/rxq.log 2>&1 \
            && say "job $n done (card $CARD)" || say "JOBFAIL $n (card $CARD)"; }
          exec 7>&-
        done
      }
      worker & worker & worker & wait
      exec 4>&- 5>&-; say "card $CARD released"
    fi
  fi
  [ -n "$(todo)" ] && { say "jobs left (other cards?): $(todo | wc -l)"; exit 0; }
else
  for o in $L/out/*; do [ -d $D/out/$(basename $o) ] || cp -al $o $D/out/; done
fi
if [ ! -f $D/level/stamp.json ]; then
  rm -rf $D/level.part
  ( flock -s 9; $PY -m rxq assemble $D/level.part $(ls -d $D/out/* | sort) --template $L/../template.json ) 9>$Q/locks/ram.lock > $D/logs/assemble.log 2>&1
  [ -f $D/level.part/stamp.json ] && mv $D/level.part $D/level && say "assembled" || { say "ASSEMBLE FAIL"; exit 1; }
fi
if [ ! -f $D/final/.done ]; then
  rm -rf $D/final.part $D/final.part.scales
  if [ -f $D/level/c9/.done ]; then
    ( flock -s 9; $PY -m rxq.post apply $D/level $D/level/c9 $D/final.part ) 9>$Q/locks/ram.lock > $D/logs/post.log 2>&1
  else cp -al $D/level $D/final.part && touch $D/final.part/.done; fi
  [ -f $D/final.part/.done ] && mv $D/final.part $D/final && rm -rf $D/final.part.scales && say "post" || { say "POST FAIL"; exit 1; }
fi
if [ ! -f $D/final.recipe ]; then
  $PY -m rxq.mix recipe $D/final > $D/final.recipe.part 2> $D/logs/recipe.log && mv $D/final.recipe.part $D/final.recipe && say "recipe" || { say "RECIPE FAIL"; exit 1; }
fi
if [ $ARM = h ] && [ ! -f $D/final/stamp-hash.json ]; then
  cp $D/final/stamp.json $D/final/stamp-hash.json
  python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d[-1]['args']['codebook']='i8'; d[-1]['w139_shim']='hash codes (exl3hash A 0xd0474b B 0x2545F491) packed in mul1i8 slot for EXL3I_BENCH_IQRH 1; original: stamp-hash.json'; json.dump(d,open(p+'.part','w'),indent=1)" $D/final/stamp.json \
    && mv $D/final/stamp.json.part $D/final/stamp.json && say "stamp shim"
fi
touch $D/DONE; say "DONE $N"
