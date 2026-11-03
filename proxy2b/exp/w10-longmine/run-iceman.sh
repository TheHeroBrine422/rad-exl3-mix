#!/bin/bash
# run-iceman.sh -- w10-longmine (ranked.md I1): mine the existing 27B long-trace evals for family x position effects.
# ICEMAN, CPU ONLY (no card, no GPU: numpy over per-row score files; < 2 GB RAM, ~1 min). Takes /root/ram.lock only
# while it runs. Copy this folder to iceman first (e.g. cat each file | ssh-iceman.sh "cat > /root/w10-longmine/...").
# Step 1 writes manifest.tsv and STOPS if any score file is unresolved: complete it by hand (tag, score file, golden,
# mix or container), or delete those lines; rerun. Step 2 runs longmine.py. Public goldens are excluded on purpose
# (public reports only, never informs a decision): calib-pool long goldens only (golden-long2*, ver3/ver4-long).
: ${RXM_WORK:=/root}; X=$(cd $(dirname $0) && pwd); O=$RXM_WORK/w10-longmine; mkdir -p $O
SD=${LM_SCORES:-$RXM_WORK/longscores}
GOLD=${LM_GOLDENS:-"$RXM_WORK/golden-long2,$RXM_WORK/golden-long2-c*"}
MIXD=${LM_MIXDIRS:-"$RXM_WORK/search/mixes,$RXM_WORK/mixes,/data/mix,$RXM_WORK/auto/mixes,/workspace/rad-exl3-mix/search/mixes"}
RADLS=${LM_RADLS:-/workspace/rad-exl3-mix/data/v2/radls.jsonl}   # add ours-/release containers' byte lists if mixes use them
exec 7>$RXM_WORK/ram.lock; flock 7
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
if [ ! -f $O/manifest.tsv ]; then
  nice -n 19 python3 $X/mkmanifest.py $RADLS $SD "$GOLD" "$MIXD" > $O/manifest.tsv
  echo "available long evals (longmix.jsonl):"; grep -o '"tag": "[^"]*"' $RXM_WORK/longmix.jsonl 2>/dev/null | sort -u | tr '\n' ' '; echo
  if grep -q "^# UNRESOLVED" $O/manifest.tsv; then
    echo "manifest $O/manifest.tsv has unresolved arms (above): complete or delete those lines, then rerun"; exit 2; fi
fi
nice -n 19 python3 $X/longmine.py $O/manifest.tsv $RADLS $O/longmine.json ${LM_REF:+--ref $LM_REF} | tee $O/result.txt
