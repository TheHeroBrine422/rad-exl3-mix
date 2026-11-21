#!/bin/bash
# score.sh TAG  -- w212: score a golden (default ds41-verify-v1; env GOLD) with bakon's radiance on box2: TP4 on all four cards (locks
# card0..3 in order). Env: POOL (host-pool-mib, 40960), SEQS (max-num-seqs, 1), HEADROOM (gpu-headroom-mib, 96).
# Output /data/q27/w212/runs/TAG/{config.yaml,report.json,report.json.rows,serve.log,time.txt}. Restartable: skips a finished run.
TAG=$1; W=/data/q27/w212; R=$W/runs/$TAG
[ -s $R/report.json ] && { echo "$TAG done"; exit 0; }
mkdir -p $R
for c in 0 1 2 3; do exec {fd}> /data/q27/locks/card$c.lock; flock $fd; done
cat > $R/config.yaml <<EOF
memory:
  gpu-headroom-mib: ${HEADROOM:-96}
models:
  ds41-official:
    model: $W/ds41-official.rad
    tp: 4
    tp-wire: exact
    placement: expert_tiered
    weights-disk-tier: true
    host-pool-mib: ${POOL:-40960}
    ngram-placement: disk
    kv-cache-dtype: bf16
    max-model-len: 9216
    max-num-seqs: ${SEQS:-1}
    max-num-batched-tokens: 8192
    checkpoint-interval: 8192
    kld-ref: ${GOLD:-$W/gold-v1}
    kld-out: $R/report.json.part
EOF
export RADIANCE_HOME=/data/radiance-ds41/build/radiance_home
S=$(date +%s)
/data/radiance-ds41/build/bin/radiance --config $R/config.yaml > $R/serve.log 2>&1
RC=$?
echo "rc=$RC seconds=$(( $(date +%s) - S ))" > $R/time.txt
if [ $RC = 0 ] && [ -s $R/report.json.part ]; then mv $R/report.json.part $R/report.json; mv $R/report.json.part.rows $R/report.json.rows; fi
