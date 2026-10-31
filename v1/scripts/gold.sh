#!/bin/bash
# gold.sh MODEL GOLDEN CORPUS OUT KVDTYPE MAXLEN [flags] -- the patched recorder, scored inline; report to OUT.json.
# Card: $CARD (default 0, the fast link), home: $RHOME (default the frozen install). Kills nothing.
M=$1 G=$2 C=$3 O=$4 KV=$5 ML=$6; shift 6
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=${RHOME:-/root/install-card/share/radiance} RADIANCE_KLD_GOLDEN=$G
export ROCR_VISIBLE_DEVICES=${CARD:-0}
rm -rf $O; s=$(date +%s)
${BIN:-/root/radiance-kldgold} --model $M --tp 1 --kv-cache-dtype $KV --max-model-len $ML --host-headroom-mib 512 --max-num-seqs 1 \
  --kld-record $O --kld-corpus $C "$@" > $O.log 2>&1
rc=$?
python3 -I /root/aggscore.py $G $O/score.f32 > $O.json 2>$O.err
echo "rc=$rc wall=$(( $(date +%s) - s ))s errs=$(grep -cE "^E |fatal" $O.log)"
