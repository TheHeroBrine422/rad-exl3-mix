#!/bin/bash
# rel207.sh (box2) -- V3 release pipeline, three loops: pack (pack207.sh, sizes in ORDER) -> verify (score final750 on card $VCARD with
# the w201 build; rows must be byte-identical to the mad mix's /data/q27/mad205/scores/<size>-mad-final750.json.rows, else mean
# KLD within 1e-5) -> upload (one file at a time, its own HF commit) to $REPO as vram/qwen3.8-27b-exl3-v3-<bpw>bpw.rad.
# Markers in /data/q27/release-v3/state/: <size>.ok / .bad / .up. Restartable. Stop: touch /data/q27/release-v3/STOP.
set -u
R=/data/q27/release-v3; Q=/data/q27; NEW=$Q/w201/rad21; G=$Q/final/golden-final750; VCARD=${VCARD:-0}
REPO=${REPO:-Hero422/qwen3.8-27b-exl3-v3-rad}
ORDER="2 2.5 2.75 3 3.25 3.5 3.75 4 4.5 5 5.5 6 6.5 7 1.5 1.75"
mkdir -p $R/state $R/verify; log() { echo "$(date -u +%H:%M:%S) $*" >> $R/rel207.log; }
pack() { for S in $ORDER; do [ -f $R/STOP ] && return; bash $R/pack207.sh $S >> $R/pack.out 2>&1; log "pack $S: $(tail -1 $R/pack.out)"; done; }
verify() { for S in $ORDER; do
  while [ ! -f $R/vram/$S.rad ]; do [ -f $R/STOP ] && return; sleep 30; done
  [ -f $R/state/$S.ok ] || [ -f $R/state/$S.bad ] && continue
  o=$R/verify/$S-final750
  flock $Q/locks/card$VCARD.lock flock -s $Q/locks/ram.lock env ROCR_VISIBLE_DEVICES=$VCARD RADIANCE_HOME=$NEW/home $NEW/install/bin/radiance \
    --model $R/vram/$S.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16448 --max-num-seqs 1 --host-headroom-mib 8192 \
    --kld-ref $G --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
  ref=$Q/mad205/scores/$S-mad-final750.json.rows
  if [ -f $o.json ] && cmp -s $o.json.rows $ref; then touch $R/state/$S.ok; log "verify $S: rows byte-identical"
  elif [ -f $o.json ] && python3 -c "
import numpy as np,sys; a=np.fromfile('$o.json.rows',np.float32).reshape(-1,4)[:,0].astype(float); b=np.fromfile('$ref',np.float32).reshape(-1,4)[:,0].astype(float)
d=abs(a.mean()-b.mean()); print('mean diff',d,'max row diff',abs(a-b).max()); sys.exit(0 if d<1e-5 else 1)" >> $R/rel207.log 2>&1; then touch $R/state/$S.ok; log "verify $S: within 1e-5"
  else touch $R/state/$S.bad; log "verify $S: FAILED"; fi
done; }
upload() { export HF_TOKEN=$(cat /root/.hf_token); hf repos create $REPO --repo-type model --exist-ok >> $R/upload.log 2>&1
  for S in $ORDER; do
  while [ ! -f $R/state/$S.ok ]; do [ -f $R/STOP ] || [ -f $R/state/$S.bad ] && break; sleep 30; done
  [ -f $R/state/$S.ok ] || continue; [ -f $R/state/$S.up ] && continue
  n=qwen3.8-27b-exl3-v3-$(printf %.2f $S)bpw.rad
  for a in 1 2 3 4 5; do
    hf upload $REPO $R/vram/$S.rad vram/$n --repo-type model --commit-message "V3 $(printf %.2f $S) bpw (mad, A8)" >> $R/upload.log 2>&1 \
      && { touch $R/state/$S.up; log "upload $S: $n done"; break; } || { log "upload $S: attempt $a failed"; sleep 120; }
  done
done; }
pack & verify & { [ -z "${NOUP:-}" ] && upload; } & wait; log "ALL DONE"
