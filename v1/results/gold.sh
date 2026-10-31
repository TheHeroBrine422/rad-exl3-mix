#!/bin/bash
# gold.sh MODEL GOLDEN CORPUS OUT KVDTYPE MAXLEN [flags] -- the patched recorder, scored inline; report to OUT.json
M=$1 G=$2 C=$3 O=$4 KV=$5 ML=$6; shift 6
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance RADIANCE_KLD_GOLDEN=$G
pkill -x radiance; pkill -f "^/root/radiance-kldgold"; while pgrep -x radiance >/dev/null || pgrep -f "^/root/radiance-kldgold" >/dev/null; do sleep 1; done
rm -rf $O; s=$(date +%s)
/root/radiance-kldgold --model $M --tp 1 --kv-cache-dtype $KV --max-model-len $ML --host-headroom-mib 512 --max-num-seqs 1 \
  --kld-record $O --kld-corpus $C "$@" > $O.log 2>&1
rc=$?
python3 -I /root/aggscore.py $G $O/score.f32 > $O.json 2>$O.err
echo "rc=$rc wall=$(( $(date +%s) - s ))s errs=$(grep -cE "^E |fatal" $O.log)"
