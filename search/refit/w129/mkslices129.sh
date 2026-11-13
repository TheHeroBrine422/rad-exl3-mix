#!/bin/bash
# mkslices129.sh -- #169's goldens on ds8 (RULES-prior.md "Data"): fit129 (calib, cap 8192), s129-c (calib, cap 4096),
# s129-x (pickx 130 pieces of xc8k-b2-c2, seed 16930, heavy cap 2048), s129 = goldmix(x, c, seed 16931).
# DOCS = draw129.py's two lines. CPU only, ram.lock shared. Restartable (finished goldens skipped).
set -eu
Q=/data/q27; W=$Q/w129; CODE=$Q/stage/code; PY=$Q/venv/bin/python; O=$W/slices
G=$Q/challenge/golden/calib; CORP=$Q/challenge/corpus/calib.jsonl; XG=$Q/goldens/xcorp/xc8k-b2-c2
exec 7>$Q/locks/ram.lock; flock -s 7
pick() { [ -f $O/$1/kld.json ] && return 0; rm -rf $O/$1.part
  nice -n 10 $PY -I $CODE/eval/golden_pick.py $G $CORP $O/$1.part $O/$1.jsonl $2 > $O/$1.log 2>&1; mv $O/$1.part $O/$1; }
pick golden-fit129 "$(sed -n 1p $W/docs129.txt)"
pick golden-s129-c "$(sed -n 2p $W/docs129.txt)"
if [ ! -f $O/golden-s129-x/kld.json ]; then
  # no corpus text for xc8k-b2-c2 here: a placeholder corpus ({} per piece; radiance scores from the golden's tokens)
  $PY -c "import sys; n=sum(1 for _ in open('$XG/corpus.meta.jsonl')); open('$W/xc8k-b2-c2.stub.jsonl','w').write('{}\n'*n)"
  ex=(); for f in $W/excl/*.json $Q/heads/slices/golden-hd122-x/kld.json; do ex+=(--exclude-orig $f); done
  rm -rf $O/golden-s129-x.part
  nice -n 10 $PY -I $CODE/eval/slices/pickx.py $XG $W/xc8k-b2-c2.stub.jsonl $XG/corpus.meta.jsonl 130 16930 \
      $O/golden-s129-x.part $O/golden-s129-x.jsonl "${ex[@]}" --cap-cat agentic,tools,long,reasoning:2048 2> $O/golden-s129-x.log
  mv $O/golden-s129-x.part $O/golden-s129-x; mv $O/golden-s129-x.part.docs $O/golden-s129-x.docs
fi
[ -f $O/golden-s129/kld.json ] || nice -n 10 $PY -I $CODE/search/v3/slices.py goldmix $O/golden-s129 16931 x=$O/golden-s129-x c=$O/golden-s129-c > $O/golden-s129.log 2>&1
echo SLICES DONE
