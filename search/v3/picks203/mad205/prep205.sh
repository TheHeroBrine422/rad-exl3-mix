#!/bin/bash
# prep205.sh (box2) -- mad-only swap check (RULES-picks am. 11-17 5:10 AM): for each frozen pick, the swapped mix = the pick's
# mix with token_embd -> emb-t-mad-K<same K> and (7.0) output.weight hd-u7.375m -> hd-u7.375a; nothing else changed.
# Input: /data/q27/mad205/base/<size>.mix (the picks' coordinator mixes). Output: /data/q27/mad205/mixes/<size>-{pick,mad}.mix
set -eu
Z=/data/q27/mad205; mkdir -p $Z/mixes
for f in $Z/base/*.mix; do
  s=$(basename $f .mix); cp $f $Z/mixes/$s-pick.mix
  sed -e 's#^token_embd.weight /data/q27/emb/trellis/emb-t-i8-K\([0-9.]*\)\.rad$#token_embd.weight /data/q27/emb/trellis/emb-t-mad-K\1.rad#' \
      -e 's#^token_embd.weight /data/q27/emb/trellis/emb-t-m1-K\([0-9.]*\)\.rad$#token_embd.weight /data/q27/emb/trellis/emb-t-mad-K\1.rad#' \
      -e 's#^output.weight /data/q27/w158/heads/hd-u\([0-9.]*\)m\.rad$#output.weight /data/q27/w158/heads/hd-u\1a.rad#' \
      $f > $Z/mixes/$s-mad.mix
  n=$(diff $Z/mixes/$s-pick.mix $Z/mixes/$s-mad.mix | grep -c '^>' || true)
  for p in $(grep -E '^(output|token_embd)\.weight ' $Z/mixes/$s-mad.mix | awk '{print $2}'); do [ -f $p ] || { echo "$s: MISSING $p"; exit 1; }; done
  echo "$s: $n line(s) changed: $(grep -E '^(output|token_embd)\.weight ' $Z/mixes/$s-mad.mix | awk '{print $2}' | xargs -n1 basename | tr '\n' ' ')"
done
