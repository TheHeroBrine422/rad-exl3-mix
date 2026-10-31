#!/bin/bash
# usage: srv.sh LOG [extra flags]
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance
LOG=$1; shift
pkill -x radiance; while pgrep -x radiance >/dev/null; do sleep 1; done
EXL3R4D_NT_DEBUG=${NTDBG:-} nohup radiance --model /root/models/q38-27b-exl3-4.00bpw.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 8192 --host-headroom-mib 512 --max-num-seqs 1 --port 4681 "$@" > $LOG 2>&1 &
for i in $(seq 1 120); do grep -q "listening on" $LOG && break; grep -qE "^E |error" $LOG && break; sleep 2; done
grep -E "kernel selection|listening|^E " $LOG
