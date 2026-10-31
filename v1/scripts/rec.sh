#!/bin/bash
# usage: rec.sh MODEL OUTDIR CORPUS LOG [extra flags]
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance
M=$1 O=$2 C=$3 L=$4; shift 4
pkill -x radiance; while pgrep -x radiance >/dev/null; do sleep 1; done
rm -rf $O
s=$(date +%s.%N)
radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 8192 --host-headroom-mib 512 --max-num-seqs 1 --kld-record $O --kld-corpus $C "$@" > $L 2>&1
echo "rc=$? wall=$(python3 -c "import time;print(round(time.time()-$s,1))")s"
