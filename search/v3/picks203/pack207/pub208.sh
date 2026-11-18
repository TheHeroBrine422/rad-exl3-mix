#!/bin/bash
# pub208.sh CARD SIZE... (box2) -- #208 public eval (published numbers; run once on the frozen, mad-only release containers, never used to
# select or tune). exl3i serves at tensor parallelism 1 only (ssm_ab / exl3i weights), so TP1, one lane per card. First try
# --max-model-len 262144 (the 2 long agentic traces whole); if the model + KV don't fit, rerun at 32768 (the long traces then do not
# finish and are left out: marked <size>.short) -- the long traces get their own pass later.
# Output /data/q27/release-v3/public/<size>.json(.rows,.log). Restartable. Stop: touch /data/q27/release-v3/STOP-pub.
set -u
R=/data/q27/release-v3; Q=/data/q27; NEW=$Q/w201/rad21; G=$Q/challenge/golden/public; C=$1; shift; mkdir -p $R/public
run() {  # run SIZE CTX
  local o=$R/public/$1
  flock $Q/locks/card$C.lock flock -s $Q/locks/ram.lock env ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$NEW/home \
    $NEW/install/bin/radiance --model $R/vram/$1.rad --tp 1 --kv-cache-dtype bf16 --max-model-len $2 --max-num-seqs 1 \
    --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $o-$2.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
}
for S in "$@"; do
  while [ ! -f $R/state/$S.ok ]; do [ -f $R/STOP-pub ] && exit 0; [ -f $R/state/$S.bad ] && break; sleep 60; done
  [ -f $R/state/$S.ok ] || continue; [ -f $R/public/$S.json ] && continue; [ -f $R/STOP-pub ] && exit 0
  t0=$(date +%s); run $S 262144 || { run $S 32768 && touch $R/public/$S.short; }
  echo "$(date -u +%H:%M:%S) public $S card $C $([ -f $R/public/$S.short ] && echo SHORT) $(( $(date +%s) - t0 )) s $(grep -o '"kld": [0-9.e-]*' $R/public/$S.json 2>/dev/null | head -1)" >> $R/pub208.log
done
