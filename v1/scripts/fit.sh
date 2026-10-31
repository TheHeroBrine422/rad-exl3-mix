#!/bin/bash
# fit.sh BPW MAXLEN -- does the server start at bf16 KV and this context? prints the KV lines
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance
pkill -x radiance; while pgrep -x radiance >/dev/null; do sleep 1; done
radiance --model /root/models/q38-27b-exl3-$1bpw.rad --tp 1 --kv-cache-dtype bf16 --max-model-len $2 --host-headroom-mib 512 --max-num-seqs 1 --port 4681 > /root/fit-$1-$2.log 2>&1 &
for i in $(seq 1 150); do grep -qE "listening on" /root/fit-$1-$2.log && break; pgrep -x radiance >/dev/null || break; sleep 2; done
pkill -x radiance
grep -qE "listening on" /root/fit-$1-$2.log && echo "$1 @ $2: STARTS" || echo "$1 @ $2: FAILS"
grep -E "^E |kv: the cache|sessions at" /root/fit-$1-$2.log | cut -c1-200 | head -4
