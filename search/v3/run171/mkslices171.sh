#!/bin/bash
# mkslices171.sh DOCS171.txt NAME:XBATCH:XSEED:MSEED ... -- the search's two-part slices on ds8 (#171; w129's
# mkslices129.sh): per NAME
#   golden-NAME-c  golden_pick of DOCS171.txt's NAME line (draw171.py: merged pool, least-used first, cap 4,096)
#   golden-NAME-x  pickx 130 pieces of night1105's XBATCH (seed XSEED, heavy categories cut to 2,048 scored positions),
#                  excluding the xcorp docs of 27b-27f-x, hd122-x, s129-x and every golden-s171*-x built before it
#   golden-NAME    slices.py goldmix (seed MSEED, x / c interleaved): the golden the evals score
# CPU only, ram.lock shared. Restartable (a finished golden is skipped, .part dirs rebuilt).
set -eu
Q=/data/q27; O=$Q/w131/slices; CODE=${CODE:-$Q/w131/code}; PY=$Q/venv/bin/python
G=$Q/challenge/golden/calib; CORP=$Q/challenge/corpus/calib.jsonl; X=$Q/goldens/xcorp
DOCS=$1; shift; mkdir -p $O
exec 7>$Q/locks/ram.lock; flock -s 7
for spec in "$@"; do
  IFS=: read NAME XB XSEED MSEED <<< "$spec"; [ -n "$MSEED" ] || { echo "bad spec $spec"; exit 1; }
  GC=$O/golden-$NAME-c GX=$O/golden-$NAME-x GM=$O/golden-$NAME
  if [ ! -f $GC/kld.json ]; then
    d=$(grep "^$NAME " $DOCS | cut -d' ' -f2-); [ -n "$d" ] || { echo "no docs for $NAME in $DOCS"; exit 1; }
    rm -rf $GC.part
    nice -n 10 $PY -I $CODE/eval/golden_pick.py $G $CORP $GC.part $GC.jsonl $d > $GC.log 2>&1; mv $GC.part $GC
  fi
  if [ ! -f $GX/kld.json ]; then
    stub=$O/$XB.stub.jsonl     # no corpus text for night1105's batches here: {} per piece (radiance scores the golden's tokens)
    [ -f $stub ] || $PY -c "n=sum(1 for _ in open('$X/$XB/corpus.meta.jsonl')); open('$stub','w').write('{}\n'*n)"
    ex=(); for f in $Q/w129/excl/*.json $Q/heads/slices/golden-hd122-x/kld.json $Q/w129/slices/golden-s129-x/kld.json; do ex+=(--exclude-orig $f); done
    for f in $O/golden-s171*-x/kld.json; do [ -f "$f" ] && ex+=(--exclude-orig $f); done
    rm -rf $GX.part
    nice -n 10 $PY -I $CODE/eval/slices/pickx.py $X/$XB $stub $X/$XB/corpus.meta.jsonl 130 $XSEED $GX.part $GX.jsonl "${ex[@]}" \
        --cap-cat agentic,tools,long,reasoning:2048 2> $GX.log
    # night1105's goldens record the bf16 model by its local path, the calib golden by its HF name (same bf16 weights);
    # goldmix requires equal "model" fields: the x part takes the calib golden's (as w129's s129-x)
    $PY -c "import json; p='$GX.part/kld.json'; m=json.load(open(p)); c=json.load(open('$G/kld.json'))['model']
assert m['model'].endswith('qwen38-27b-bf16'), m['model']; m['model'] = c; json.dump(m, open(p, 'w'))"
    mv $GX.part $GX; mv $GX.part.docs $GX.docs
  fi
  [ -f $GM/kld.json ] || nice -n 10 $PY -I $CODE/search/v3/slices.py goldmix $GM $MSEED x=$GX c=$GC > $GM.log 2>&1
  echo "$(date -u +%FT%TZ) $NAME: c $(grep -c . $GC.jsonl) docs, x $(grep -c . $GX.jsonl) pieces -> $GM"
done
echo SLICES DONE
