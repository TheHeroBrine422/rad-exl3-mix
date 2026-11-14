#!/bin/bash
# evals181.sh CARD JOB... (w154, #181) -- served-path judgement jobs on this host, one card (HIP index) under its lock
# for every score (eval181.py takes conv + ram exclusive for packs / mixes, card + ram shared for scores). JOB =
#   MIX:TAG[:CKPT]   eval181.py TAG on vl/MIX with served/MIX.json; CKPT = a run's ckpt dir name under runs/RUN (as
#                    RUN/ckpt-xxx), none = the control; slices c (verify27-w114) + x (golden-x181g)
#   gate:MIX         the identity gate: MIX's control with the base embedding scored on golden-s171d (ds8 only)
# Restartable (eval181 skips finished steps). Results in eval/MIX/TAG/result.json.
set -u
C=$1; shift
cd /data/q27/w154 && source env.sh
H=/data/q27/multi/host.json
SL=(--slice c=$Q/w125/slices/golden-verify27-w114 --slice x=$W/slices/golden-x181g)
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" | tee -a $W/evals181.log; }
for job in "$@"; do
  IFS=: read kind a b <<< "$job"
  if [ "$kind" = gate ]; then
    mix=$a; mkdir -p eval/$mix
    $PY $CODE/v3/qad181/eval181.py ctl-gate vl/$mix eval/$mix --host $H --served served/$mix.json --emb base \
      --slice s171d=$W/slices/golden-s171d --card $C --card-lock $Q/locks/card$C.lock --keep-pack >> eval/$mix/ctl-gate.out 2>&1 \
      || { log "gate $mix FAILED"; continue; }
    ref=$W/ref147/v3s171-2.5-block-00626-s171d.json.rows
    if cmp -s eval/$mix/ctl-gate/score-s171d.json.rows $ref; then log "GATE PASS $mix: control rows on s171d byte-identical to w147's B rows"
    else log "GATE DIFF $mix: rows differ from w147's B ($(python3 -c "
import numpy as np; a=np.fromfile('eval/$mix/ctl-gate/score-s171d.json.rows',np.float32).reshape(-1,4)[:,0]; b=np.fromfile('$ref',np.float32).reshape(-1,4)[:,0]
print(f'mean {a.mean():.6f} vs {b.mean():.6f}, max |d| {abs(a-b).max():.2e}, rows equal {(a==b).mean():.4f}')"))"; fi
    continue
  fi
  mix=$kind tag=$a ck=${b:-}; mkdir -p eval/$mix
  sc=(); [ -n "$ck" ] && sc=(--scales $W/runs/$ck)
  [ -n "$ck" ] && [ ! -f $W/runs/$ck/.done ] && [ ! -f eval/$mix/$tag/mix.rad ] && { log "$mix $tag: $ck not done"; continue; }
  log "$mix $tag (card $C) ${ck:+scales $ck}"
  $PY $CODE/v3/qad181/eval181.py $tag vl/$mix eval/$mix --host $H --served served/$mix.json "${sc[@]}" "${SL[@]}" \
    --card $C --card-lock $Q/locks/card$C.lock >> eval/$mix/$tag.out 2>&1 || { log "$mix $tag FAILED (eval/$mix/$tag.out)"; continue; }
  log "$mix $tag: $(python3 -c "import json; r=json.load(open('eval/$mix/$tag/result.json')); print(', '.join(f\"{k} {v['kld']:.5f} ({v['positions']} pos)\" for k, v in r['slices'].items()))")"
  [ -n "$ck" ] && rm -rf eval/$mix/$tag/level      # the healed level copy (~8-20 GB); scales stay in runs/
done
