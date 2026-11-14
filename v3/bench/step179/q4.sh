#!/bin/bash
# q4.sh -- card 1, after the merge with qc-exl3-v3-mad (w152, #179):
#  1. AQL-stamp timing (stamp.sh; no sync between ops): d0 / d8 decode (ctx 0 / 8k), p8 the 8k prefill; homes base /
#     step2 on the mul1i8 packs (w138), mbase (qc-exl3-v3-mad e054e71) / mstep (b02aa9a) on the mad packs (w150 / w151)
#  2. KLD rows mbase vs mstep on the mad packs (def, c64); base-714 vs step2-714 on K4.125 mul1i8 (def)
#  3. gdec (greedy decode, logprobs) mbase / mstep on the mad packs
#  4. e2e rounds 4, 5 on the mad packs (mbase, mstep interleaved)
# Every unit skips when its output exists. Log logs/q.log. (w152)
W=/data/q27/w152; cd $W; Q=/data/q27; P8=$Q/w137/prompts/ctx-8k.txt
M1="v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8"
MAD="$Q/w150/packs/q27v3-635bac02-k2.5-mad-a8.rad $Q/w150/packs/q27v3-144dd6e1-k4.125-mad-a8.rad"
mkdir -p res/st; say() { echo "$(date -u +%T) q4 $*" >> logs/q.log; }
st() {   # st HOMENAME PACKPATH
  local h=$1 m=$2 pn=$(basename $2 .rad)
  bash tools/stamp.sh 1 $W/b10$h/home $m res/st/$h-$pn-d0
  bash tools/stamp.sh 1 $W/b10$h/home $m res/st/$h-$pn-d8 $P8
  NTOK=1 STAMP_N=1 bash tools/stamp.sh 1 $W/b10$h/home $m res/st/$h-$pn-p8 $P8
  say "stamp $h $pn"
}
for p in $M1; do st base $Q/w138/packs/$p.rad; st step2 $Q/w138/packs/$p.rad; done
for m in $MAD; do st mbase $m; st mstep $m; done
for m in $MAD; do bash tools/kld.sh 1 $m mbase-def mstep-def mbase-c64 mstep-c64 >> logs/kld.log 2>&1; done
bash tools/kld.sh 1 v3-k4.125-r16k4-a8 base-714-def step2-714-def >> logs/kld.log 2>&1
say "kld"
for m in $MAD; do bash tools/gdec.sh 1 $m mbase $W/b10mbase/home; bash tools/gdec.sh 1 $m mstep $W/b10mstep/home; done
say "gdec"
for r in 4 5; do ROWS="mbase=$W/b10mbase/home mstep=$W/b10mstep/home" bash tools/e2.sh 1 $r $MAD; done
say DONE
