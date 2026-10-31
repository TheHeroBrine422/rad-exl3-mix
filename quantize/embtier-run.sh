#!/bin/bash
# embtier-run.sh PLAN... -- per PLAN: a reflinked copy of the sc4.00-e4 trunk with a bf16 embedding, fake-quantised
# tiered (embtier.py), queued on calib-A as vt-emb[-PFX]-<PLAN> (q.prio; vt- tags stay out of the fit). 3 preps at a time.
# env VT_RANK (counts file, default rank-sq-counts.npy = calib search+qad), VT_PFX (tag infix, e.g. g for the guard ranking).
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final}; export RXM_WORK RXM_MODELS
V=/data/vt; cd $V; mkdir -p emb
if [ ! -f emb/base.rad ]; then
  echo "token_embd.weight $RXM_MODELS/q38-27b-bf16.rad" > emb/base.mix
  $RXM_WORK/hproxy/radmix emb/base.rad.part $RXM_MODELS/q38-27b-sc4.00-e4.rad emb/base.mix 2> emb/radmix.log && mv emb/base.rad.part emb/base.rad || exit 1
fi
OFF=$(PATH=$RXM_WORK/install-card/bin:$PATH rad-info -v emb/base.rad 2>&1 | grep " token_embd.weight " | grep -o "@[0-9]*" | tr -d @)
[ -n "$OFF" ] || { echo "no offset"; exit 1; }
one() {
  local p=$1 t=vt-emb${VT_PFX:+-$VT_PFX}-${1//[:,]/_}; local o=$V/emb/$t.rad
  cp --reflink=always emb/base.rad $o.part && nice /data/venv-exl3/bin/python -I $V/embtier.py $o.part $OFF $V/${VT_RANK:-rank-sq-counts.npy} "$p" --write >> emb/prep.log 2>&1 \
    && mv $o.part $o || { echo "$p FAIL" >> emb/prep.log; return; }
  ( exec 8>>$RXM_WORK/auto/q.claimlock; flock 8; echo "$t $o - golden-calA corpus/calA.jsonl" >> $RXM_WORK/auto/q.prio
    [ -d $RXM_WORK/golden-ml ] && echo "${t/vt-emb/vt-ml} $o - golden-ml corpus/ml.jsonl" >> $RXM_WORK/auto/q.prio )
}
for p in "$@"; do one "$p" & while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 5; done; done
wait; echo "EMBPREPDONE ${VT_PFX:-sq}" >> emb/prep.log
