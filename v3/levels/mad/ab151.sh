#!/bin/bash
# ab151.sh [CARDS] -- (ds8) #177-03 a/b donors with the mad codebook: abtrellis-mad.py (w142's abtrellis.py, codebook mad)
# at K4 and K5 (#178-02's T4, #178-03's T5) -> /data/q27/mad/ab/K{4,5}/ (rxq job outputs: t-*.safetensors with codebook mad
# + the settings' a / b) and ab-T{4,5}.safetensors (the decoded bf16 a / b, for simulated arms). The served donor containers
# (w147's donor147.sh: recipe line `blk.*.ssm_ab.weight exl3i ... codebook=mad`) need #177-02's importer: not built here.
# First free card of CARDS (default "6 5 4 3 2 1 0", never 7), card lock + ram.lock shared. Restartable (rxq skips finished
# tensors). Process name `bash /data/q27/mad/tools/ab151.sh`. (w151)
set -u; Q=/data/q27; O=$Q/mad/ab; T=$Q/mad/tools; mkdir -p $O; exec >> $O/ab151.log 2>&1
say() { echo "$(date -u +%T) $*"; }
source $Q/tools/env-ds8.sh
[ -f $O/ab-T4.safetensors ] && [ -f $O/ab-T5.safetensors ] && { say "done earlier"; exit 0; }
while :; do for c in ${1:-6 5 4 3 2 1 0}; do [ $c = 7 ] && continue; exec 5>$Q/locks/card$c.lock; flock -n 5 && break 2; exec 5>&-; done; sleep 30; done
exec 6>$Q/locks/ram.lock; flock -s 6; say "card $c"
cd $Q/stage/code/pipeline && ROCR_VISIBLE_DEVICES=$c nice -n 5 $PY $T/abtrellis-mad.py $Q/models/qwen38-27b-bf16 $Q/challenge/hessians/calib $O -K 4,5 --codebook mad \
  && say "AB151 DONE: $(ls $O/K4/t-*.safetensors | wc -l) + $(ls $O/K5/t-*.safetensors | wc -l) tensors" || say "AB151 FAIL"
