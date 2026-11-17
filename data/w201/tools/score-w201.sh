#!/bin/bash
# score-w201.sh BUILD MIX GOLDEN OUTBASE [PROGRESS] -- one s195c2 eval on box2 card 3 (ROCR/HIP index 3 = rocm-smi GPU[0]) with
# the coordinator's score command from /data/q27/srch183/host.json (score_cmd, or score_cmd_progress when PROGRESS is given),
# written as OUTBASE.json / .json.rows (the coordinator's layout). Holds card3.lock exclusive and ram.lock shared (evals' rule).
# Skips a finished OUTBASE.json (rerun-safe; a killed run leaves only .part files, which are overwritten).
# BUILD is a radiance prefix holding install/bin/radiance and home/ (rad21 = this w201 build; w173/rad10 = the old build).
set -u
B=$1; MIX=$2; GOLD=$3; O=$4; PROG=${5:-}
[ -f "$O.json" ] && { echo "skip $O.json"; exit 0; }
mkdir -p "$(dirname "$O")"
PA=(); [ -n "$PROG" ] && { rm -f "$PROG" "$PROG.kl"; PA=(--kld-progress "$PROG"); }   # a progress file is appended to: start it empty
t0=$(date +%s)
flock /data/q27/locks/card3.lock flock -s /data/q27/locks/ram.lock \
  env ROCR_VISIBLE_DEVICES=3 RADIANCE_HOME="$B/home" "$B/install/bin/radiance" --model "$MIX" --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16448 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref "$GOLD" --kld-out "$O.json.part" "${PA[@]}" \
    > "$O.log" 2>&1 \
  && mv "$O.json.part.rows" "$O.json.rows" && mv "$O.json.part" "$O.json"
rc=$?
echo "$(basename "$O") rc=$rc secs=$(( $(date +%s) - t0 ))"
exit $rc
