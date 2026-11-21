#!/bin/bash
# score-port.sh TAG -- w213: score a golden (default ds41-verify-v1; env GOLD) with mikoskinen's EXL3 2.25 bpw container on the PORT (exl3-v3 + bakon's DS4.1-F,
# /data/q27/w213/src/build), otherwise exactly w212b's score-mik.sh (same YAML). Env BUILD: another build dir.
# (box2, TP4 on all four cards, locks card0..3 in order). bakon's ds41/this-machine/tp4-exl3.yaml settings (even expert split = no
# RADIANCE_EXPERT_SPLIT, host-pool 48 GiB, expert_tiered, ngram disk), but: tp-wire exact, kv bf16, one sequence, no speculation / DSpark,
# no prefix-resume, thinking off. Env: POOL (host-pool-mib, 49152), SEQS (1), HEADROOM (96), TP (4), CARDS (0 1 2 3).
# Output /data/q27/w213/runs/TAG/{config.yaml,report.json,report.json.rows,serve.log,time.txt}. Restartable: skips a finished run.
TAG=$1; W=/data/q27/w212; R=/data/q27/w213/runs/$TAG; B=${BUILD:-/data/q27/w213/src/build}
[ -s $R/report.json ] && { echo "$TAG done"; exit 0; }
mkdir -p $R
CARDS=${CARDS:-0 1 2 3}
for c in $CARDS; do exec {fd}> /data/q27/locks/card$c.lock; flock $fd; done
cat > $R/config.yaml <<YEOF
cards: [$(echo $CARDS | sed 's/ /,/g')]
memory:
  gpu-headroom-mib: ${HEADROOM:-96}
models:
  ds41-mik225:
    model: $W/mik225/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
    tp: ${TP:-4}
    tp-wire: exact
    placement: expert_tiered
    host-pool-mib: ${POOL:-49152}
    ngram-placement: disk
    kv-cache-dtype: bf16
    max-model-len: 9216
    max-num-seqs: ${SEQS:-1}
    max-num-batched-tokens: 2048
    checkpoint-interval: 8192
    chat-template-kwargs: {"enable_thinking":false}
    kld-ref: ${GOLD:-$W/gold-v1}
    kld-out: $R/report.json.part
YEOF
export RADIANCE_HOME=$B/radiance_home
S=$(date +%s)
$B/bin/radiance --config $R/config.yaml > $R/serve.log 2>&1
RC=$?
echo "rc=$RC seconds=$(( $(date +%s) - S ))" > $R/time.txt
if [ $RC = 0 ] && [ -s $R/report.json.part ]; then mv $R/report.json.part $R/report.json; mv $R/report.json.part.rows $R/report.json.rows; fi
