#!/bin/bash
# mkx177.sh -- RULES-177 amendment 1's fresh xcorp part on ds8: golden-h177-x = pickx 130 pieces of night1105's
# xc8k-b2-c2 (seed 17701, agentic / tools / long / reasoning cut to 2,048 scored positions), excluding every earlier x part
# (27b-27f via w129/excl, hd122-x, s129-x, s171a-d-x, golden-t-x); the x branch of search/v3/run171/mkslices171.sh
# (same pickx call, stub corpus, model field = the calib golden's). CPU only, ram.lock shared. Out /data/q27/w139s3/slices/.
# (w139, #177)
set -eu
Q=/data/q27; O=$Q/w139s3/slices; CODE=$Q/w131/code; PY=$Q/venv/bin/python; X=$Q/goldens/xcorp; XB=xc8k-b2-c2
G=$Q/challenge/golden/calib; GX=$O/golden-h177-x
mkdir -p $O; exec 7>$Q/locks/ram.lock; flock -s 7
[ -f $GX/kld.json ] && { echo "built"; exit 0; }
stub=$O/$XB.stub.jsonl
[ -f $stub ] || $PY -c "n=sum(1 for _ in open('$X/$XB/corpus.meta.jsonl')); open('$stub','w').write('{}\n'*n)"
ex=(); for f in $Q/w129/excl/*.json $Q/heads/slices/golden-hd122-x/kld.json $Q/w129/slices/golden-s129-x/kld.json \
    $Q/w131/slices/golden-s171a-x/kld.json $Q/w131/slices/golden-s171b-x/kld.json $Q/w131/slices/golden-s171c-x/kld.json \
    $Q/w131/slices/golden-s171d-x/kld.json $Q/w121/chk/multi/golden-t-x/kld.json $Q/split/chk/multi/golden-t-x/kld.json; do
  [ -f $f ] || { echo "missing exclusion list $f"; exit 1; }; ex+=(--exclude-orig $f); done
rm -rf $GX.part
nice -n 10 $PY -I $CODE/eval/slices/pickx.py $X/$XB $stub $X/$XB/corpus.meta.jsonl 130 17701 $GX.part $GX.jsonl "${ex[@]}" \
    --cap-cat agentic,tools,long,reasoning:2048 2> $GX.log
$PY -c "import json; p='$GX.part/kld.json'; m=json.load(open(p)); c=json.load(open('$G/kld.json'))['model']
assert m['model'].endswith('qwen38-27b-bf16'), m['model']; m['model'] = c; json.dump(m, open(p, 'w'))"
mv $GX.part $GX; mv $GX.part.docs $GX.docs
echo "$(date -u +%FT%TZ) h177-x: $(grep -c . $GX.jsonl) pieces; $(cat $GX.log | head -c 300)"
