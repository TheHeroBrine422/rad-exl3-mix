#!/bin/bash
# mkfinal.sh -- final-pick slice parts on ds8 (lead, 11-13; w145 plan): c3 (123 reused verify docs, cap 8192), x (210 fresh
# pieces of xc8k-b2-c2, s171 / s173 x excluded), then final750 = goldmix(x, c1 = golden-ver4 (copied, not scored), c3).
set -eu
Q=/data/q27; O=$Q/final; CODE=$Q/lead/code; PY=$Q/venv/bin/python
G=$Q/challenge/golden/calib; CORP=$Q/challenge/corpus/calib.jsonl; X=$Q/goldens/xcorp; XB=xc8k-b2-c2
mkdir -p $O; exec 7>$Q/locks/ram.lock; flock -s 7
GC=$O/golden-final-c3 GX=$O/golden-final-x GM=$O/golden-final750
if [ ! -f $GC/kld.json ]; then rm -rf $GC.part; nice -n 10 $PY -I $CODE/eval/golden_pick.py $G $CORP $GC.part $GC.jsonl 71:8192 83:8192 185:8192 227:8192 301:8192 327:8192 331:8192 359:8192 409:8192 497:8192 576:8192 587:8192 588:8192 591:8192 715:8192 873:8192 887:8192 925:8192 1099:8192 1122:8192 1157:8192 1164:8192 1271:8192 1361:8192 1381:8192 1397:8192 1411:8192 1427:8192 1455:8192 1461:8192 1468:8192 1491:8192 1499:8192 1527:8192 1537:8192 1611:8192 1621:8192 1681:8192 1707:8192 1714:8192 1789:8192 1853:8192 1867:8192 1983:8192 2041:8192 2167:8192 2189:8192 2194:8192 2195:8192 2209:8192 2215:8192 2255:8192 2256:8192 2318:8192 2349:8192 2381:8192 2391:8192 2399:8192 2439:8192 2445:8192 2451:8192 2487:8192 2533:8192 2545:8192 2671:8192 2787:8192 2849:8192 2913:8192 2971:8192 3007:8192 3046:8192 3054:8192 3142:8192 3157:8192 3243:8192 3323:8192 3351:8192 3369:8192 3409:8192 3461:8192 3491:8192 3516:8192 3517:8192 3527:8192 3549:8192 3560:8192 3568:8192 3631:8192 3707:8192 3727:8192 3749:8192 3791:8192 3797:8192 3819:8192 3841:8192 3851:8192 3867:8192 3916:8192 3933:8192 3945:8192 3981:8192 4007:8192 4014:8192 4067:8192 4075:8192 4175:8192 4193:8192 4223:8192 4277:8192 4286:8192 4367:8192 4374:8192 4465:8192 4475:8192 4479:8192 4494:8192 4509:8192 4617:8192 4647:8192 4651:8192 4667:8192 4683:8192 4755:8192 > $GC.log 2>&1; mv $GC.part $GC; fi
if [ ! -f $GX/kld.json ]; then
  stub=$O/$XB.stub.jsonl; [ -f $stub ] || $PY -c "n=sum(1 for _ in open('$X/$XB/corpus.meta.jsonl')); open('$stub','w').write('{}\n'*n)"
  ex=(); for f in $Q/w129/excl/*.json $Q/heads/slices/golden-hd122-x/kld.json $Q/w129/slices/golden-s129-x/kld.json $Q/w131/slices/golden-s17[13]*-x/kld.json; do [ -f "$f" ] && ex+=(--exclude-orig $f); done
  rm -rf $GX.part
  nice -n 10 $PY -I $CODE/eval/slices/pickx.py $X/$XB $stub $X/$XB/corpus.meta.jsonl 210 7501 $GX.part $GX.jsonl "${ex[@]}" --cap-cat agentic,tools,long,reasoning:2048 2> $GX.log
  $PY -c "import json; p='$GX.part/kld.json'; m=json.load(open(p)); c=json.load(open('$G/kld.json'))['model']
assert m['model'].endswith('qwen38-27b-bf16'), m['model']; m['model'] = c; json.dump(m, open(p, 'w'))"
  mv $GX.part $GX; mv $GX.part.docs $GX.docs
fi
[ -f $GM/kld.json ] || nice -n 10 $PY -I $CODE/search/v3/slices.py goldmix $GM 7502 x=$GX c1=$Q/goldens/golden-ver4 c3=$GC > $GM.log 2>&1
echo "$(date -u +%FT%TZ) final750: $($PY -c "import json;m=json.load(open('$GM/kld.json'));print(len(m['docs']),'docs',m['rows'],'rows')")"
