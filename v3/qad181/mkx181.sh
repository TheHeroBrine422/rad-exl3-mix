#!/bin/bash
# mkx181.sh (w154, #181, ds8): the fresh xcorp judging part golden-x181 = 150 pieces of night1105's xc8k-b2-c2 (pickx.py,
# seed 181, heavy categories cut to 2,048 scored positions as the search's x parts), excluding every xcorp DOCUMENT that
# any earlier slice's x part used (w129 excl, hd122-x, s129-x, s171* / s173* x, final-x) AND every document with a
# piece in the #181 training chunks (xq-c00..39, xc8k-b3-c0..4: other pieces of the same document are never judged).
# CPU only. Restartable.
set -eu
Q=/data/q27; W=$Q/w154; O=$W/slices; CODE=$W/code; PY=$Q/venv/bin/python; X=$Q/goldens/xcorp; XB=xc8k-b2-c2
G=$Q/challenge/golden/calib; GX=$O/golden-x181; mkdir -p $O
[ -f $GX/kld.json ] && { echo "x181 done"; exit 0; }
$PY -c "
import json, glob
docs = set()
for f in sorted(glob.glob('$W/gold/xq-c*/corpus.meta.jsonl') + glob.glob('$W/gold/xc8k-b3-c*/corpus.meta.jsonl')):
    docs |= {json.loads(l)['doc'] for l in open(f)}
json.dump({'docs': [{'orig': d} for d in sorted(docs)]}, open('$O/train181-docs.json', 'w'))
print(len(docs), 'training xcorp documents excluded')"
stub=$O/$XB.stub.jsonl
[ -f $stub ] || $PY -c "n=sum(1 for _ in open('$X/$XB/corpus.meta.jsonl')); open('$stub','w').write('{}\n'*n)"
ex=(--exclude-orig $O/train181-docs.json)
for f in $Q/w129/excl/*.json $Q/heads/slices/golden-hd122-x/kld.json $Q/w129/slices/golden-s129-x/kld.json \
         $Q/w131/slices/golden-s17*-x/kld.json $Q/final/golden-final-x/kld.json; do [ -f "$f" ] && ex+=(--exclude-orig $f); done
echo "exclusions: ${#ex[@]} args"
rm -rf $GX.part
nice -n 10 $PY -I $CODE/eval/slices/pickx.py $X/$XB $stub $X/$XB/corpus.meta.jsonl 150 181 $GX.part $GX.jsonl "${ex[@]}" \
    --cap-cat agentic,tools,long,reasoning:2048 2> $GX.log
$PY -c "import json; p='$GX.part/kld.json'; m=json.load(open(p)); c=json.load(open('$G/kld.json'))['model']
assert m['model'].endswith('qwen38-27b-bf16'), m['model']; m['model'] = c; json.dump(m, open(p, 'w'))
print(len(m['docs']), 'pieces', m['rows'], 'rows')"
mv $GX.part $GX; mv $GX.part.docs $GX.docs
echo "X181 DONE"
