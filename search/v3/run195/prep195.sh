#!/bin/bash
# prep195.sh BATCH... -- on ds8 (CPU, nice): pull the finished box2 goldens of #195 (xn8k-b195, xc8k-b2-c3, xc8k-b2-c4) over the 10 GbE
# link, make their board gen masks (genmask181.py make) and their xsupply rows (xsupply186.py: labels, used flags), and link them
# into the xview the draw / build read. The ssh line to box2 is read from the lead's push-box2.sh (not kept in the repo).
set -eu
Q=/data/q27; W=$Q/w180; C=$W/code; PY="nice -n 19 $Q/venv/bin/python3 -I"; TOK=$Q/models/qwen38-27b-bf16
eval "$(sed -n 3p $Q/srch183/push-box2.sh | sed 's/^set -e; //')"       # B (ssh command), H (box2)
mkdir -p $W/xnew $W/xview $W/strata
for b in "$@"; do
  $B $H "test -f $W/batch/gold/$b/.done"
  rsync -a --partial -e "$B" $H:$W/batch/gold/$b $W/xnew/
  [ -f $W/xnew/$b/gm.u8 ] || { $PY $C/v3/qad181/genmask181.py make $W/xnew/$b $W/xnew/$b/gm.u8.part $TOK; mv $W/xnew/$b/gm.u8.part $W/xnew/$b/gm.u8; }
  mkdir -p $W/xview/$b; for f in kld.json tokens.i32 top.bin rows.f32 corpus.meta.jsonl gm.u8; do ln -sfn $W/xnew/$b/$f $W/xview/$b/$f; done
done
S=$Q/w131/slices; S6=$Q/w164/slices
GX="$(ls -d $S/golden-s17*-x $S/golden-t183-x $S6/golden-s186?-x $S6/golden-t186n-x $Q/w169/slices/golden-s190?-x $Q/w176/slices/golden-s190?-x $Q/w179/slices/golden-s190?-x $Q/final/golden-final-x)"   # not golden-final750: its c parts carry calib "orig" numbers (its x part = final-x)
$PY $C/search/v3/run186w/xsupply186.py $TOK $W/strata/xsupply195.jsonl $W/xnew $GX
( cd $W/xnew && md5sum */kld.json */rows.f32 */tokens.i32 */top.bin */gm.u8 ) > $W/strata/md5-batch-ds8.txt
echo PREP195 DONE
