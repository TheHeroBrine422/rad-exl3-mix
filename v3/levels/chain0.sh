#!/bin/bash
# chain0.sh -- card 0 overnight (2026-10-31): Phase-B settings, then the V3 quarter/half levels K2.25-4.75.
#  1. gaintest.py: g_scale multiplier per family, layers 3/10/43 (attention, GDN, late attention), K2/4/6
#  2. damptest.py at high K (K6/8; the turboderp-wins-high hypothesis: over-damping), layers 3/10/43
#  3. waits for /data/v3levels/settings.env (written after review: OURQ_ARGS="--sigma-reg-in .. --gain .."), and for
#     the fractional-K smoke test to pass; then each level, trunk only (heads/embedding are separate options),
#     exllamav3-format exports in /data/v3levels/K<k> (no radiance import for fractional K yet).
# Locks: card0.lock per job (evals and other card-0 jobs interleave), ram.lock shared for the sweeps.
: ${RXM_WORK:=/root}; export RXM_WORK
W=/data/v3levels PY=/data/venv-exl3/bin/python L=/data/v3levels/chain0.log
mkdir -p $W; cd $RXM_WORK
log() { echo "$(date -u +%H:%M) $*" | tee -a $L; }
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# after the YAQA work that needs both cards (gate 2c, then the gate-2e collection)
until grep -q "^DONE" /data/yaqa/full64.log 2>/dev/null || ! pgrep -f "^/bin/bash ./gate2e.sh" > /dev/null; do sleep 60; done
run() { ( exec 9>$RXM_WORK/card0.lock 7>$RXM_WORK/ram.lock; flock 9; flock -s 7; "$@" ); }
if ! grep -q GAINDONE $W/gain.log 2>/dev/null; then
  run $PY -I ourq/gaintest.py $W/gain.jsonl 3,10,43 2,4,6 0.85,0.92,1.0,1.08,1.16 --device 0 > $W/gain.log 2>&1
  echo "GAINDONE rc=$?" >> $W/gain.log; log "gain sweep done: $(wc -l < $W/gain.jsonl) rows"
fi
if ! grep -q DAMPDONE $W/damp.log 2>/dev/null; then
  run $PY -I ourq/damptest.py $W/damp.jsonl 3,10,43 6,8 0.025,0.005,0.001 --device 0 > $W/damp.log 2>&1
  echo "DAMPDONE rc=$?" >> $W/damp.log; log "high-K damping sweep done: $(wc -l < $W/damp.jsonl) rows"
fi
log "waiting for $W/settings.env and the fractional smoke test"
DEADLINE=$(date -u -d "07:30" +%s); [ $DEADLINE -lt $(date +%s) ] && DEADLINE=$((DEADLINE + 86400))
until [ -f $W/settings.env ] && grep -q "FRACSMOKE OK" /data/exl3frac/smoke.log 2>/dev/null; do
  grep -q "FRACSMOKE FAIL" /data/exl3frac/smoke.log 2>/dev/null && { log "smoke test FAILED: levels not started"; exit 1; }
  if [ ! -f $W/settings.env ] && [ $(date +%s) -gt $DEADLINE ]; then   # nobody decided by ~2:30 AM CT: the documented default
    echo 'OURQ_ARGS="--sigma-reg-in 0.005"' > $W/settings.env; log "no settings by the deadline: default --sigma-reg-in 0.005"; fi
  sleep 120; done
. $W/settings.env
log "levels with: $OURQ_ARGS"
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
if ! grep -q '"key"' $W/trial/proxy.jsonl 2>/dev/null; then          # one layer (3: attention inputs) with the settings
  rm -rf $W/trial
  ( exec 9>$RXM_WORK/card0.lock; flock 9
    $PY -I ourq/ourq.py -K 3.25 -o $W/trial --layers 3-3 --device 0 --frac-ext /data/exl3frac/src $OURQ_ARGS 2>&1 | grep -v -e Warning -e "cpu = _conv" ) > $W/trial.log
  [ -f $W/trial/q-L03.safetensors ] || { log "trial layer FAILED (see trial.log): levels not started"; exit 1; }
  log "trial layer ok: $(grep -o '"sigma_reg": [0-9.e-]*' $W/trial/proxy.jsonl | sort | uniq -c | tr '\n' ' ')"
fi
for K in ${LEVELS:-3.25 2.75 3.75 3.5 2.5 3 4.25 4 2.25 4.5 4.75 2 5 5.5 6}; do   # 2.5-4 bpw first; integers rebuilt with the int8 codebook
  [ -f $W/K$K/q-L63.safetensors ] && continue
  s=$(date +%s)
  ( exec 9>$RXM_WORK/card0.lock; flock 9
    $PY -I ourq/ourq.py -K $K -o $W/K$K --device 0 --frac-ext /data/exl3frac/src $OURQ_ARGS 2>&1 | grep -v -e Warning -e "cpu = _conv" ) > $W/K$K.log
  [ -f $W/K$K/q-L63.safetensors ] && log "K$K done in $(( ($(date +%s) - s) / 60 )) min, $(du -sh $W/K$K | cut -f1)" || log "K$K FAIL"
done
log "chain0 done"
